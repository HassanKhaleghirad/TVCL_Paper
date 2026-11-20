#!/usr/bin/env python3
"""
TVCL (EAD/CSSL + sparse spatial-temporal attention + MAR) on InTAS-style VeReMi logs.
Protocol: VEHICLE-DISJOINT split of senders (train/val/test = 60/20/20, stratified by sender label), repeated
over seeds. Model selection (epoch, threshold) uses the validation vehicles only; test vehicles are touched once.
Output rows follow detection_runs.csv (table,dataset,condition,model,seed,tp,fp,tn,fn,auc,pr_auc).

  python tvcl_intas.py --data /mnt/user-data/uploads --seeds 0 1 2 ... --out detection_runs_intas.csv
"""
import argparse, math, os, csv, copy, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from feats import load, FEATS
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data
def build(df, seed, T=12, val=0.2, test=0.2, fixed_test=False):
    """Vehicle-disjoint split + snapshot tensors. Statistics are fitted on TRAIN senders only."""
    senders = np.array(sorted(df.send.unique()))
    lab = df.groupby("send").att.max().reindex(senders).values
    tr_s, te_s = train_test_split(senders, test_size=test, stratify=lab, random_state=(12345 if fixed_test else seed))
    lab_tr = df.groupby("send").att.max().reindex(tr_s).values
    tr_s, va_s = train_test_split(tr_s, test_size=val / (1 - test), stratify=lab_tr, random_state=seed)
    sid = {s: i for i, s in enumerate(senders)}
    split = np.zeros(len(senders), int)                      # 0 train, 1 val, 2 test
    split[[sid[s] for s in va_s]] = 1; split[[sid[s] for s in te_s]] = 2
    mu = df[df.send.isin(tr_s)][FEATS].mean(); sd = df[df.send.isin(tr_s)][FEATS].std().replace(0, 1)
    Z = ((df[FEATS] - mu) / sd).clip(-5, 5)
    b = np.minimum((np.argsort(np.argsort(df.t.values, kind="stable"), kind="stable") * T // len(df)), T - 1)
    N = len(senders)
    X = np.zeros((N, T, len(FEATS)), np.float32); M = np.zeros((N, T), bool); Ys = np.zeros((N, T), np.float32)
    g = pd.DataFrame(Z.values, columns=FEATS); g["n"] = df.send.map(sid).values; g["b"] = b; g["att"] = df.att.values
    for (n, bb), r in g.groupby(["n", "b"]):
        X[n, bb] = r[FEATS].mean().values; M[n, bb] = True; Ys[n, bb] = r.att.mean()
    A = np.zeros((T, N, N), np.float32)                      # co-reception counts
    recv = df.recv.values
    for bb in range(T):
        d = g[g.b == bb]; r = recv[d.index]
        for rv in np.unique(r):
            ids = np.unique(d.n.values[r == rv])
            A[bb][np.ix_(ids, ids)] += 1
    A[:, np.arange(N), np.arange(N)] = 0
    W = np.log1p(A) / max(np.log1p(A).max(), 1e-9)
    lab_snap = (Ys > 0.5).astype(np.float32)
    trust = 1.0 - Ys                                         # proxy trust target (from attack annotations)
    return dict(X=torch.tensor(X), M=torch.tensor(M), A=torch.tensor(A > 0), W=torch.tensor(W),
                y=torch.tensor(lab_snap), trust=torch.tensor(trust), split=torch.tensor(split), senders=senders)


# ----------------------------------------------------------------------------- model
class TVCL(nn.Module):
    def __init__(s, nf, d=64, S=4, k=3, Da=5, mar=True, ead=True, drop=0.1, skip=False):
        super().__init__()
        s.d, s.S, s.k, s.Da, s.mar, s.ead, s.skip = d, S, k, Da, mar, ead, skip
        s.f1 = nn.Sequential(nn.Linear(nf, d), nn.ReLU(), nn.Linear(d, d))          # f_theta (node part)
        s.fn = nn.Linear(d, d)                                                      # f_theta (neighbour part)
        s.mal = nn.Linear(3 * d, 1)
        s.Ws = nn.Linear(d, d, bias=False); s.a1 = nn.Parameter(torch.randn(d) * .1); s.a2 = nn.Parameter(torch.randn(d) * .1)
        s.P = nn.Linear(d, d); s.u = nn.Parameter(torch.randn(d) * .1)               # phi(x,w) = P x + u w
        s.Wq, s.Wk, s.Wv, s.Wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.drop = nn.Dropout(drop)
        s.head = nn.Sequential(nn.Linear(d * (2 if skip else 1), d), nn.ReLU(), nn.Linear(d, 1))

    def pos(s, T, dev):
        p = torch.arange(T, device=dev)[:, None]; i = torch.arange(0, s.d, 2, device=dev)
        pe = torch.zeros(T, s.d, device=dev); pe[:, 0::2] = torch.sin(p / 10000 ** (i / s.d)); pe[:, 1::2] = torch.cos(p / 10000 ** (i / s.d))
        return pe

    def embed(s, X, A, M):                                   # EAD encoder per snapshot -> e [N,T,d]
        e = s.f1(X)
        Af = A.float().permute(1, 0, 2)                      # [N,T,N]
        deg = Af.sum(-1, keepdim=True).clamp(min=1)
        agg = torch.einsum("ntm,mtd->ntd", Af, e) / deg
        return (e + s.fn(agg)) * M[..., None]

    def views(s, e, M):                                      # causal short / long views for every (v,t)
        N, T, d = e.shape; k = s.k
        cs = torch.cat([torch.zeros(N, 1, d, device=e.device), torch.cumsum(e * M[..., None], 1)], 1)
        cn = torch.cat([torch.zeros(N, 1, device=e.device), torch.cumsum(M.float(), 1)], 1)
        short = torch.zeros_like(e); long = torch.zeros_like(e); has = torch.zeros(N, T, dtype=torch.bool, device=e.device)
        for t in range(T):
            lo = max(t + 1 - k, 0)
            short[:, t] = (cs[:, t + 1] - cs[:, lo]) / (cn[:, t + 1] - cn[:, lo]).clamp(min=1)[:, None]
            h = t - k
            if h < 0: continue
            Eh, Mh = e[:, :h + 1], M[:, :h + 1]
            with torch.no_grad():
                En = Eh.masked_fill(~Mh[..., None], float("nan"))
                med = torch.nanmedian(En, dim=1).values                                   # coordinate-wise median
                dv = 1 - F.cosine_similarity(Eh, med[:, None, :], dim=-1)
                dv = dv.masked_fill(~Mh, float("nan"))
                gam = torch.nanquantile(dv, 0.95, dim=1, keepdim=True)
                keep = (dv <= gam) & Mh
            kf = keep.float()[..., None]
            long[:, t] = (kf * Eh).sum(1) / kf.sum(1).clamp(min=1)
            has[:, t] = keep.any(1)
        long = torch.where(has[..., None], long, short)
        return short, long, has

    def forward(s, X, A, W, M):
        N, T, _ = X.shape
        e = s.embed(X, A, M)
        short, long, has = s.views(e, M)
        if s.ead:
            mh = torch.sigmoid(s.mal(torch.cat([short, long, short - long], -1))).squeeze(-1) * M
        else:
            long = short; mh = torch.zeros(N, T, device=X.device)
        x = short                                                                           # x_v^{(t)}
        Wx = s.Ws(x)
        l1 = (Wx * s.a1).sum(-1).T; l2 = (Wx * s.a2).sum(-1).T                               # [T,N]
        logit = F.leaky_relu(l1[:, :, None] + l2[:, None, :], 0.2)                           # [T,N,N]
        eye = torch.eye(N, dtype=torch.bool, device=X.device)[None]
        mask = (A | eye) & M.T[:, None, :] & M.T[:, :, None]
        alpha = torch.softmax(logit.masked_fill(~mask, -1e9), -1) * mask
        if s.mar:
            g = alpha * (1 - mh.detach()).T[:, None, :]
            alpha = g / (g.sum(-1, keepdim=True) + 1e-8)
        Wf = torch.where(eye, torch.ones_like(W), W)
        msg = torch.einsum("tij,jtd->itd", alpha, s.P(x)) + torch.einsum("tij,tij->ti", alpha, Wf).T[..., None] * s.u
        h = F.elu(msg) * M[..., None]
        ht = s.drop(h + s.pos(T, X.device)[None])
        q, k_, v = (f(ht).view(N, T, s.S, -1) for f in (s.Wq, s.Wk, s.Wv))
        sc = torch.einsum("nisd,njsd->nsij", q, k_) / math.sqrt(q.shape[-1])
        idx = torch.arange(T, device=X.device); di = idx[:, None] - idx[None, :]
        allow = (di >= 0) & ((di <= s.Da) | (idx[None, :] == 0))                              # causal + local + global(first)
        allow = allow[None, None] & M[:, None, None, :]
        att = torch.softmax(sc.masked_fill(~allow, -1e9), -1) * allow
        z = s.Wo(torch.einsum("nsij,njsd->nisd", att, v).reshape(N, T, -1)) * M[..., None]
        yh = torch.sigmoid(s.head(torch.cat([z, short], -1) if s.skip else z)).squeeze(-1)
        return dict(trust=yh, mal=mh, short=short, long=long, has=has, M=M)


def cssl_loss(o, M, nodes, beta=0.1):
    """InfoNCE between each node's last-k view and its robust history (own last present snapshot)."""
    T = M.shape[1]; last = (M.long() * torch.arange(T)[None]).argmax(1)
    ok = [n for n in nodes if o["has"][n, last[n]]]
    if len(ok) < 4: return o["short"].sum() * 0
    sh = F.normalize(torch.stack([o["short"][n, last[n]] for n in ok]), dim=-1)
    lg = F.normalize(torch.stack([o["long"][n, last[n]] for n in ok]), dim=-1)
    return F.cross_entropy(sh @ lg.T / beta, torch.arange(len(ok)))


def run(D, seed, mar=True, cssl=True, ead=True, epochs=250, pre=80, patience=40, lam=(1.0, 0.1, 0.5), k=3, d=64, drop=0.1, wd=1e-4, skip=False, ret_val=False):
    torch.manual_seed(seed); np.random.seed(seed)
    X, A, W, M, y, tr, sp = D["X"], D["A"], D["W"], D["M"], D["y"], D["trust"], D["split"]
    m_tr = M & (sp == 0)[:, None]; m_va = M & (sp == 1)[:, None]; m_te = M & (sp == 2)[:, None]
    net = TVCL(X.shape[-1], d=d, mar=mar, ead=ead, k=k, drop=drop, skip=skip)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
    tr_nodes = [i for i in range(len(sp)) if sp[i] == 0]
    pos = y[m_tr].mean().clamp(1e-3, 1 - 1e-3); wpos = (1 - pos) / pos
    if cssl and ead:
        for _ in range(pre):                                                 # phase 1: label-free pretraining
            opt.zero_grad(); loss = cssl_loss(net(X, A, W, M), M, tr_nodes); loss.backward(); opt.step()
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):                                                 # phase 2: joint fine-tuning
        net.train(); opt.zero_grad(); o = net(X, A, W, M)
        wt = torch.where(y[m_tr] > 0.5, wpos, torch.ones(()))
        l_tr = (wt * (o["trust"][m_tr] - tr[m_tr]) ** 2).mean()
        l_mal = F.binary_cross_entropy(o["mal"][m_tr].clamp(1e-6, 1 - 1e-6), y[m_tr], weight=wt) if ead else 0 * l_tr
        l_cs = cssl_loss(o, M, tr_nodes) if (cssl and ead) else 0 * l_tr
        (lam[0] * l_tr + lam[1] * l_cs + lam[2] * l_mal).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            ov = net(X, A, W, M); sv = (1 - ov["trust"][m_va]).numpy(); yv = y[m_va].numpy()
        a = roc_auc_score(yv, sv) if 0 < yv.sum() < len(yv) else 0.5
        if a > best + 1e-4: best, best_state, bad = a, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        o = net(X, A, W, M)
    sv, yv = (1 - o["trust"][m_va]).numpy(), y[m_va].numpy()
    fpr, tpr, th = roc_curve(yv, sv); thr = th[np.argmax(tpr - fpr)]            # Youden on validation
    st, yt = (1 - o["trust"][m_te]).numpy(), y[m_te].numpy()
    r = conf(yt, st, thr)
    return (r, best) if ret_val else r


def conf(y, s, thr):
    p = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    ap = average_precision_score(y, s) if y.sum() > 0 else float("nan")
    return int(tp), int(fp), int(tn), int(fn), float(auc), float(ap)


def baseline(D, kind):
    X, M, y, sp = D["X"], D["M"], D["y"], D["split"]
    m = {i: M & (sp == i)[:, None] for i in range(3)}
    mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0) if kind == "GBM" \
        else LogisticRegression(max_iter=1000, class_weight="balanced")
    mdl.fit(X[m[0]].numpy(), y[m[0]].numpy())
    sv = mdl.predict_proba(X[m[1]].numpy())[:, 1]; fpr, tpr, th = roc_curve(y[m[1]].numpy(), sv)
    return conf(y[m[2]].numpy(), mdl.predict_proba(X[m[2]].numpy())[:, 1], th[np.argmax(tpr - fpr)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/datset"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--condition", default="constantPositionOffset"); ap.add_argument("--out", default="detection_runs_intas.csv")
    ap.add_argument("--ablation", action="store_true"); ap.add_argument("--fixed-test", action="store_true"); ap.add_argument("--cfg", default="{}"); ap.add_argument("--epochs", type=int, default=250)
    a = ap.parse_args()
    df = load(a.data); new = not os.path.exists(a.out)
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new: w.writerow("table,dataset,condition,model,seed,tp,fp,tn,fn,auc,pr_auc".split(","))
    jobs = [("attack", "TVCL", dict())]
    if a.ablation:
        jobs = [("ablation", "TVCL (Full)", dict()), ("ablation", "w/o MAR", dict(mar=False)),
                ("ablation", "w/o CSSL", dict(cssl=False)), ("ablation", "w/o EAD", dict(ead=False, cssl=False))]
    for sd in a.seeds:
        D = build(df, sd, fixed_test=a.fixed_test)
        for tab, name, kw in jobs:
            r = run(D, sd, epochs=a.epochs, **{**eval(a.cfg), **kw}); w.writerow([tab, "intas", a.condition if tab == "attack" else "default", name, sd, *r]); f.flush()
            print(sd, name, "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f" % r)
        if not a.ablation:
            for b in ("GBM", "LogReg"):
                r = baseline(D, b); w.writerow(["attack", "intas", a.condition, b + " (msg features)", sd, *r]); print(sd, b, "auc=%.3f" % r[4])
