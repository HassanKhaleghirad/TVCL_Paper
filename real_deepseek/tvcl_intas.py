#!/usr/bin/env python3
"""
TVCL (EAD/CSSL + sparse spatial-temporal attention + MAR) on InTAS-style VeReMi logs.

Protocol (vehicle-disjoint, stratified, repeated over seeds):
  * split senders 60/20/20 into train / val / test.
  * Statistics fitted on TRAIN senders only.
  * Model selection (epoch, threshold) on VAL only; TEST touched once.
  * Snapshot binning: each vehicle's messages are partitioned into T=12 bins
    by MESSAGE ORDER (not absolute time), so every vehicle with >= 3 messages
    has a real temporal sequence and CSSL/EAD/MAR are active.

Ablation flags (now truly independent):
  * mar=False  -> no trust-weighted attention (uniform aggregation)
  * cssl=False -> no contrastive short/long view loss
  * ead=False  -> no EAD module (long view replaced by short view; mh=0)
  * w/o all temporal = mar=False, cssl=False, ead=False
"""
import argparse, math, os, csv, copy, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, confusion_matrix)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from real_deepseek.OK_MAR.feats import load, FEATS
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data
def build(df, seed, T=12, val=0.2, test=0.2, fixed_test=False,
          min_snaps_assert=True):
    """
    Vehicle-disjoint split + snapshot tensors.

    Snapshot binning: for each vehicle, its messages (sorted by t) are split
    into T contiguous bins by message index, so every vehicle with >= 1 message
    fills at least one bin, and vehicles with >= 3 messages get a real sequence.
    """
    senders = np.array(sorted(df.send.unique()))
    lab = df.groupby("send").att.max().reindex(senders).values.astype(int)

    tr_s, te_s = train_test_split(
        senders, test_size=test, stratify=lab,
        random_state=(12345 if fixed_test else seed))
    lab_tr = df.groupby("send").att.max().reindex(tr_s).values.astype(int)
    tr_s, va_s = train_test_split(
        tr_s, test_size=val / (1 - test), stratify=lab_tr, random_state=seed)

    sid = {s: i for i, s in enumerate(senders)}
    split = np.zeros(len(senders), int)
    split[[sid[s] for s in va_s]] = 1
    split[[sid[s] for s in te_s]] = 2

    mu = df[df.send.isin(tr_s)][FEATS].mean()
    sd = df[df.send.isin(tr_s)][FEATS].std().replace(0, 1)
    Z = ((df[FEATS] - mu) / sd).clip(-5, 5)

    # ---- message-order binning per vehicle (NOT global time) ----
    n = df.send.map(sid).values.astype(np.int64)
    order = (df.sort_values(["send", "t"], kind="stable")
               .groupby("send").cumcount().reindex(df.index).values)
    cnt = df.groupby("send").send.transform("size").reindex(df.index).values
    # bin = floor(order * T / cnt), ensures 0..T-1 and contiguous
    b = np.minimum((order * T // np.maximum(cnt, 1)), T - 1).astype(np.int64)

    N = len(senders)
    X = np.zeros((N, T, len(FEATS)), np.float32)
    M = np.zeros((N, T), bool)
    Ys = np.zeros((N, T), np.float32)
    Trust = np.zeros((N, T), np.float32)

    g = pd.DataFrame(Z.values, columns=FEATS)
    g["n"] = n
    g["b"] = b
    g["att"] = df.att.values.astype(np.float32)
    for (nn_, bb), r in g.groupby(["n", "b"]):
        X[nn_, bb] = r[FEATS].mean().values
        M[nn_, bb] = True
        Ys[nn_, bb] = r["att"].mean()
        # trust target = 1 - attack fraction in the bin
        Trust[nn_, bb] = 1.0 - r["att"].mean()

    # ---- co-reception adjacency (per snapshot) ----
    A = np.zeros((T, N, N), np.float32)
    recv = df.recv.values
    for bb in range(T):
        d = g[g.b == bb]
        if len(d) == 0:
            continue
        r = recv[d.index]
        for rv in np.unique(r):
            ids = np.unique(d.n.values[r == rv])
            if len(ids) < 2:
                continue
            A[bb][np.ix_(ids, ids)] += 1
    A[:, np.arange(N), np.arange(N)] = 0
    W = np.log1p(A) / max(np.log1p(A).max(), 1e-9)

    lab_snap = (Ys > 0.5).astype(np.float32)

    if min_snaps_assert:
        per = M.sum(1)
        frac3 = float((per >= 3).mean())
        frac4 = float((per >= 4).mean())
        print("[build] snapshots/vehicle: mean=%.2f  median=%.0f  "
              "frac(>=3)=%.1f%%  frac(>=4)=%.1f%%"
              % (per.mean(), float(np.median(per)), 100 * frac3, 100 * frac4))
        if frac3 < 0.3:
            print("  WARNING: <30%% of vehicles have >=3 snapshots; CSSL/EAD "
                  "will be weak.  Increase T (bins) or check input data.")

    return dict(X=torch.tensor(X), M=torch.tensor(M),
                A=torch.tensor(A > 0), W=torch.tensor(W),
                y=torch.tensor(lab_snap), trust=torch.tensor(Trust),
                split=torch.tensor(split), senders=senders)


# ----------------------------------------------------------------------------- model
class TVCL(nn.Module):
    def __init__(s, nf, d=64, S=4, k=3, Da=5, mar=True, ead=True, cssl=True,
                 drop=0.1, skip=False):
        super().__init__()
        s.d, s.S, s.k, s.Da = d, S, k, Da
        s.mar, s.ead, s.cssl, s.skip = mar, ead, cssl, skip
        s.f1 = nn.Sequential(nn.Linear(nf, d), nn.ReLU(), nn.Linear(d, d))
        s.fn = nn.Linear(d, d)
        s.mal = nn.Linear(3 * d, 1)
        s.Ws = nn.Linear(d, d, bias=False)
        s.a1 = nn.Parameter(torch.randn(d) * .1)
        s.a2 = nn.Parameter(torch.randn(d) * .1)
        s.P = nn.Linear(d, d)
        s.u = nn.Parameter(torch.randn(d) * .1)
        s.Wq, s.Wk, s.Wv, s.Wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.drop = nn.Dropout(drop)
        s.head = nn.Sequential(
            nn.Linear(d * (2 if skip else 1), d), nn.ReLU(), nn.Linear(d, 1))

    def pos(s, T, dev):
        p = torch.arange(T, device=dev)[:, None]
        i = torch.arange(0, s.d, 2, device=dev)
        pe = torch.zeros(T, s.d, device=dev)
        pe[:, 0::2] = torch.sin(p / 10000 ** (i / s.d))
        pe[:, 1::2] = torch.cos(p / 10000 ** (i / s.d))
        return pe

    def embed(s, X, A, M):
        e = s.f1(X)
        Af = A.float().permute(1, 0, 2)   # [N,T,N]
        deg = Af.sum(-1, keepdim=True).clamp(min=1)
        agg = torch.einsum("ntm,mtd->ntd", Af, e) / deg
        return (e + s.fn(agg)) * M[..., None]

    def views(s, e, M):
        """
        short: causal last-k mean.
        long : robust (median + 5%-quantile keep) history of everything before t.
        has[t]: whether a long view (t >= k) exists at time t.
        """
        N, T, d = e.shape
        k = s.k
        cs = torch.cat([torch.zeros(N, 1, d, device=e.device),
                        torch.cumsum(e * M[..., None], 1)], 1)
        cn = torch.cat([torch.zeros(N, 1, device=e.device),
                        torch.cumsum(M.float(), 1)], 1)
        short = torch.zeros_like(e)
        long = torch.zeros_like(e)
        has = torch.zeros(N, T, dtype=torch.bool, device=e.device)
        for t in range(T):
            lo = max(t + 1 - k, 0)
            short[:, t] = (cs[:, t + 1] - cs[:, lo]) / \
                          (cn[:, t + 1] - cn[:, lo]).clamp(min=1)[:, None]
            h = t - k
            if h < 0:
                continue
            Eh, Mh = e[:, :h + 1], M[:, :h + 1]
            # only consider vehicles with at least 2 present snapshots in history
            pres = Mh.sum(1) >= 2
            if not pres.any():
                continue
            with torch.no_grad():
                En = Eh.masked_fill(~Mh[..., None], float("nan"))
                med = torch.nanmedian(En, dim=1).values
                dv = 1 - F.cosine_similarity(Eh, med[:, None, :], dim=-1)
                dv = dv.masked_fill(~Mh, float("nan"))
                gam = torch.nanquantile(dv, 0.95, dim=1, keepdim=True)
                keep = (dv <= gam) & Mh & pres[:, None]
            kf = keep.float()[..., None]
            long[:, t] = (kf * Eh).sum(1) / kf.sum(1).clamp(min=1)
            has[:, t] = keep.any(1)
        long = torch.where(has[..., None], long, short)
        return short, long, has

    def forward(s, X, A, W, M):
        N, T, _ = X.shape
        e = s.embed(X, A, M)
        short, long, has = s.views(e, M)

        # EAD: per-snapshot attack probability; masked by `has` so it
        # contributes NOTHING when the long view does not exist.
        if s.ead:
            mh = torch.sigmoid(
                s.mal(torch.cat([short, long, short - long], -1))
            ).squeeze(-1) * M * has.float()
        else:
            long = short
            mh = torch.zeros(N, T, device=X.device)

        # sparse spatial attention (uses short view only)
        x = short
        Wx = s.Ws(x)
        l1 = (Wx * s.a1).sum(-1).T
        l2 = (Wx * s.a2).sum(-1).T
        logit = F.leaky_relu(l1[:, :, None] + l2[:, None, :], 0.2)
        eye = torch.eye(N, dtype=torch.bool, device=X.device)[None]
        mask = (A | eye) & M.T[:, None, :] & M.T[:, :, None]
        alpha = torch.softmax(logit.masked_fill(~mask, -1e9), -1) * mask

        # MAR: trust-weighted aggregation (only if enabled)
        if s.mar:
            g = alpha * (1 - mh.detach()).T[:, None, :]
            alpha = g / (g.sum(-1, keepdim=True) + 1e-8)

        Wf = torch.where(eye, torch.ones_like(W), W)
        msg = torch.einsum("tij,jtd->itd", alpha, s.P(x)) \
              + torch.einsum("tij,tij->ti", alpha, Wf).T[..., None] * s.u
        h = F.elu(msg) * M[..., None]

        # temporal self-attention (causal, local, global-first)
        ht = s.drop(h + s.pos(T, X.device)[None])
        q, k_, v = (f(ht).view(N, T, s.S, -1) for f in (s.Wq, s.Wk, s.Wv))
        sc = torch.einsum("nisd,njsd->nsij", q, k_) / math.sqrt(q.shape[-1])
        idx = torch.arange(T, device=X.device)
        di = idx[:, None] - idx[None, :]
        allow = (di >= 0) & ((di <= s.Da) | (idx[None, :] == 0))
        allow = allow[None, None] & M[:, None, None, :]
        att = torch.softmax(sc.masked_fill(~allow, -1e9), -1) * allow
        z = s.Wo(torch.einsum("nsij,njsd->nisd", att, v).reshape(N, T, -1)) * M[..., None]

        yh = torch.sigmoid(
            s.head(torch.cat([z, short], -1) if s.skip else z)
        ).squeeze(-1)
        return dict(trust=yh, mal=mh, short=short, long=long, has=has, M=M)


def cssl_loss(o, M, nodes, beta=0.1):
    """InfoNCE between each node's last-k view and its robust long view."""
    T = M.shape[1]
    last = (M.long() * torch.arange(T)[None]).argmax(1)
    ok = [n for n in nodes
          if o["has"][n, last[n]] and M[n, last[n]]]
    if len(ok) < 4:
        return o["short"].sum() * 0
    sh = F.normalize(torch.stack([o["short"][n, last[n]] for n in ok]), dim=-1)
    lg = F.normalize(torch.stack([o["long"][n, last[n]] for n in ok]), dim=-1)
    return F.cross_entropy(sh @ lg.T / beta, torch.arange(len(ok)))


def run(D, seed, mar=True, cssl=True, ead=True, epochs=250, pre=40,
        patience=40, lam=(1.0, 0.1, 0.5), k=3, d=64, drop=0.1, wd=1e-4,
        skip=False, ret_val=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, A, W, M, y, tr, sp = (D["X"], D["A"], D["W"], D["M"],
                              D["y"], D["trust"], D["split"])
    m_tr = M & (sp == 0)[:, None]
    m_va = M & (sp == 1)[:, None]
    m_te = M & (sp == 2)[:, None]

    net = TVCL(X.shape[-1], d=d, mar=mar, ead=ead, cssl=cssl,
               k=k, drop=drop, skip=skip)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
    tr_nodes = [i for i in range(len(sp)) if sp[i] == 0]
    pos = y[m_tr].mean().clamp(1e-3, 1 - 1e-3)
    wpos = (1 - pos) / pos

    # CSSL pretraining (only if CSSL active AND EAD active, since long view exists only with EAD)
    if cssl and ead:
        for _ in range(pre):
            opt.zero_grad()
            loss = cssl_loss(net(X, A, W, M), M, tr_nodes)
            loss.backward()
            opt.step()

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        o = net(X, A, W, M)
        wt = torch.where(y[m_tr] > 0.5, wpos, torch.ones(()))
        l_tr = (wt * (o["trust"][m_tr] - tr[m_tr]) ** 2).mean()
        l_mal = (F.binary_cross_entropy(
            o["mal"][m_tr].clamp(1e-6, 1 - 1e-6), y[m_tr], weight=wt)
            if ead else 0 * l_tr)
        l_cs = cssl_loss(o, M, tr_nodes) if (cssl and ead) else 0 * l_tr
        (lam[0] * l_tr + lam[1] * l_cs + lam[2] * l_mal).backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            ov = net(X, A, W, M)
            sv = (1 - ov["trust"][m_va]).numpy()
            yv = y[m_va].numpy()
        a = roc_auc_score(yv, sv) if 0 < yv.sum() < len(yv) else 0.5
        if a > best + 1e-4:
            best, best_state, bad = a, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        o = net(X, A, W, M)

    sv, yv = (1 - o["trust"][m_va]).numpy(), y[m_va].numpy()
    fpr, tpr, th = roc_curve(yv, sv)
    thr = th[np.argmax(tpr - fpr)]
    st, yt = (1 - o["trust"][m_te]).numpy(), y[m_te].numpy()
    r = conf(yt, st, thr)
    return (r, best) if ret_val else r


def conf(y, s, thr):
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    ap = average_precision_score(y, s) if y.sum() > 0 else float("nan")
    return int(tp), int(fp), int(tn), int(fn), float(auc), float(ap)


def baseline(D, kind):
    X, M, y, sp = D["X"], D["M"], D["y"], D["split"]
    m = {i: M & (sp == i)[:, None] for i in range(3)}
    mdl = (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                          random_state=0)
           if kind == "GBM"
           else LogisticRegression(max_iter=1000, class_weight="balanced"))
    mdl.fit(X[m[0]].numpy(), y[m[0]].numpy())
    sv = mdl.predict_proba(X[m[1]].numpy())[:, 1]
    fpr, tpr, th = roc_curve(y[m[1]].numpy(), sv)
    return conf(y[m[2]].numpy(), mdl.predict_proba(X[m[2]].numpy())[:, 1],
                th[np.argmax(tpr - fpr)])


# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/dataset")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--condition", default="constantPositionOffset")
    ap.add_argument("--out", default="detection_runs_intas.csv")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--fixed-test", action="store_true")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--epochs", type=int, default=250)
    a = ap.parse_args()

    df = load(a.data)
    print("[load] messages=%d  vehicles=%d  attack rate=%.3f"
          % (len(df), df.send.nunique(), df.att.mean()))

    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow("table,dataset,condition,model,seed,tp,fp,tn,fn,auc,pr_auc".split(","))

    if a.ablation:
        jobs = [
            ("ablation", "TVCL (Full)",      dict()),
            ("ablation", "w/o MAR",          dict(mar=False)),
            ("ablation", "w/o CSSL",         dict(cssl=False)),
            ("ablation", "w/o EAD",          dict(ead=False)),
            ("ablation", "w/o all temporal", dict(mar=False, cssl=False, ead=False)),
        ]
    else:
        jobs = [("attack", "TVCL", dict())]

    for sd in a.seeds:
        D = build(df, sd, fixed_test=a.fixed_test,
                  min_snaps_assert=(sd == a.seeds[0]))
        for tab, name, kw in jobs:
            r = run(D, sd, epochs=a.epochs, **{**eval(a.cfg), **kw})
            w.writerow([tab, "intas",
                        a.condition if tab == "attack" else "default",
                        name, sd, *r])
            f.flush()
            print(sd, name, "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f" % r)

        if not a.ablation:
            for b in ("GBM", "LogReg"):
                r = baseline(D, b)
                w.writerow(["attack", "intas", a.condition,
                            b + " (msg features)", sd, *r])
                print(sd, b, "auc=%.3f" % r[4])

    f.close()