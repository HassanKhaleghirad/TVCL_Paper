#!/usr/bin/env python3
"""
TVCL on the REAL InTAS/VeReMi-Extension JSON logs (one veh_*.json per receiver). No synthetic data.

Protocol (identical for every variant and baseline; nothing is tuned on test vehicles)
  * all *.json in --data are read; messages are grouped by sender_id (the real vehicle identity);
  * VEHICLE-DISJOINT split of senders 60/20/20 (train/val/test), stratified by sender label, new split per seed;
  * features are standardised with TRAIN statistics only; epoch (val AUC) and threshold (Youden on val) use val only;
  * test vehicles are evaluated once per run.

What is different from tvcl_intas.py (and why)
  * InTAS vehicles are present for only ~50 s, so 12 snapshots over the whole 2-hour log gave ~1 snapshot per vehicle
    and the k=3 short / long views never existed (CSSL loss was identically 0). Here the timeline is cut into
    segments of T snapshots of dt seconds (default 48 x 5 s = 240 s); a node is a (vehicle, segment) tracklet and each
    segment has its own co-reception graph. The model code (EAD, CSSL, MAR, sparse causal attention) is unchanged.
  * Features are built from the BSM fields only (no simulator noise fields). The logged distance_to_road_edge ('road')
    is kept by default because the paper uses it; run with --no-road to see how much of the result depends on it.
  * Extra baseline "GBM (+own-history)" so that any benefit of temporal modelling is compared fairly.

  python tvcl_real.py --data dataset --diag-only
  python tvcl_real.py --data dataset --seeds 0 1 2 3 4 5 6 7 8 9 --out runs_real.csv
  python tvcl_real.py --data dataset --seeds 0 1 2 3 4 5 6 7 8 9 --no-road --out runs_real_noroad.csv
Results are written exactly as obtained; report all variants, including those that do not beat Full.
"""
import argparse, glob, os, csv, copy, json, math, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

FEATS_ALL = ["dist", "spd", "dspd", "acl", "dhed", "hbear", "delay", "road"]


# ----------------------------------------------------------------------------- data (numpy / pandas only)
def _xy(s):
    a = s.split(","); return float(a[0]), float(a[1])


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def load_dir(path, time_scale=1e9, max_files=None):
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    if max_files: files = files[:max_files]
    rows = []
    for fn in files:
        rv = os.path.splitext(os.path.basename(fn))[0]
        try:
            with open(fn) as f: msgs = json.load(f)
        except Exception as e:
            print("skip", fn, e); continue
        for m in msgs:
            s, r = m["sender"], m["receiver"]
            sx, sy = _xy(s["pos"]); rx, ry = _xy(r["pos"]); dx, dy = sx - rx, sy - ry
            bearing = math.degrees(math.atan2(dx, dy)) % 360.0
            rows.append((m["sender_id"], rv, m["rcvTime"] / time_scale, int(m["attacker"]), math.hypot(dx, dy),
                         s["spd"], s["spd"] - r["spd"], s["acl"], abs(_wrap(s["hed"] - r["hed"])),
                         abs(_wrap(s["hed"] - bearing)), (m["rcvTime"] - m["sendTime"]) / time_scale,
                         s.get("distance_to_road_edge", np.nan)))
    df = pd.DataFrame(rows, columns=["send", "recv", "t", "att"] + FEATS_ALL)
    df["road"] = df["road"].fillna(df["road"].median())
    return df.reset_index(drop=True)


def make_graph(df, feats, dt, T):
    """Segments of T snapshots x dt seconds; node = (sender, segment); co-reception graph per snapshot."""
    b = np.floor((df.t.values - df.t.values.min()) / dt).astype(np.int64)
    d = df[["send", "recv", "att"] + feats].copy(); d["seg"] = b // T; d["loc"] = b % T
    d["rc"] = pd.factorize(d.recv)[0]
    node = d.groupby(["send", "seg", "loc"])[feats + ["att"]].mean().reset_index()
    segs = []
    for sg, nd in node.groupby("seg"):
        senders = nd.send.unique(); idx = {s: i for i, s in enumerate(senders)}; n = len(senders)
        ii = nd.send.map(idx).values; ll = nd["loc"].values.astype(int)
        X = np.zeros((n, T, len(feats)), np.float32); M = np.zeros((n, T), bool); Ys = np.zeros((n, T), np.float32)
        X[ii, ll] = nd[feats].values; M[ii, ll] = True; Ys[ii, ll] = nd.att.values
        ds = d[d.seg == sg][["send", "loc", "rc"]].drop_duplicates()
        ds["n"] = ds.send.map(idx).values; ds = ds.sort_values(["loc", "rc"])
        loc, rc, nn_ = ds["loc"].values, ds.rc.values, ds.n.values
        key = loc.astype(np.int64) * (int(rc.max()) + 1) + rc
        cuts = np.flatnonzero(np.diff(key)) + 1
        A = np.zeros((T, n, n), np.float32)
        for gi, gl in zip(np.split(nn_, cuts), np.split(loc, cuts)):
            if len(gi) > 1: A[gl[0]][np.ix_(gi, gi)] += 1
        A[:, np.arange(n), np.arange(n)] = 0
        segs.append(dict(senders=senders, X=X, M=M, Ys=Ys, A=A > 0, L=np.log1p(A).astype(np.float16)))
    mx = max(float(s["L"].max()) for s in segs)
    for s in segs: s["W"] = (s["L"].astype(np.float32) / max(mx, 1e-9)); del s["L"]
    return segs


def prepare(segs, df, seed, val=0.2, test=0.2):
    senders = np.array(sorted(df.send.unique()))
    lab = df.groupby("send").att.max().reindex(senders).values
    tr, te = train_test_split(senders, test_size=test, stratify=lab, random_state=seed)
    lab_tr = df.groupby("send").att.max().reindex(tr).values
    tr, va = train_test_split(tr, test_size=val / (1 - test), stratify=lab_tr, random_state=seed)
    role = {**{s: 0 for s in tr}, **{s: 1 for s in va}, **{s: 2 for s in te}}
    sps = [np.array([role[x] for x in s["senders"]]) for s in segs]
    Xtr = np.concatenate([s["X"][s["M"] & (sp == 0)[:, None]] for s, sp in zip(segs, sps)])
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    P = []
    for s, sp in zip(segs, sps):
        Z = np.clip((s["X"] - mu) / sd, -5, 5) * s["M"][..., None]
        P.append(dict(X=torch.tensor(Z, dtype=torch.float32), M=torch.tensor(s["M"]), A=torch.tensor(s["A"]),
                      W=torch.tensor(s["W"]), y=torch.tensor((s["Ys"] > 0.5).astype(np.float32)),
                      trust=torch.tensor(1.0 - s["Ys"]), sp=torch.tensor(sp)))
    return P


# ----------------------------------------------------------------------------- model (same as tvcl_intas.py)
class TVCL(nn.Module):
    def __init__(s, nf, d=64, S=4, k=3, Da=5, mar=True, ead=True, drop=0.1, skip=False):
        super().__init__()
        s.d, s.S, s.k, s.Da, s.mar, s.ead, s.skip = d, S, k, Da, mar, ead, skip
        s.f1 = nn.Sequential(nn.Linear(nf, d), nn.ReLU(), nn.Linear(d, d)); s.fn = nn.Linear(d, d)
        s.mal = nn.Linear(3 * d, 1)
        s.Ws = nn.Linear(d, d, bias=False); s.a1 = nn.Parameter(torch.randn(d) * .1); s.a2 = nn.Parameter(torch.randn(d) * .1)
        s.P = nn.Linear(d, d); s.u = nn.Parameter(torch.randn(d) * .1)
        s.Wq, s.Wk, s.Wv, s.Wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.drop = nn.Dropout(drop)
        s.head = nn.Sequential(nn.Linear(d * (2 if skip else 1), d), nn.ReLU(), nn.Linear(d, 1))

    def pos(s, T, dev):
        p = torch.arange(T, device=dev)[:, None]; i = torch.arange(0, s.d, 2, device=dev)
        pe = torch.zeros(T, s.d, device=dev); pe[:, 0::2] = torch.sin(p / 10000 ** (i / s.d)); pe[:, 1::2] = torch.cos(p / 10000 ** (i / s.d))
        return pe

    def embed(s, X, A, M):
        e = s.f1(X); Af = A.float().permute(1, 0, 2)
        deg = Af.sum(-1, keepdim=True).clamp(min=1)
        return (e + s.fn(torch.einsum("ntm,mtd->ntd", Af, e) / deg)) * M[..., None]

    def views(s, e, M):
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
                med = torch.nanmedian(En, dim=1).values
                dv = (1 - F.cosine_similarity(Eh, med[:, None, :], dim=-1)).masked_fill(~Mh, float("nan"))
                gam = torch.nanquantile(dv, 0.95, dim=1, keepdim=True)
                keep = (dv <= gam) & Mh
            kf = keep.float()[..., None]
            long[:, t] = (kf * Eh).sum(1) / kf.sum(1).clamp(min=1)
            has[:, t] = keep.any(1)
        return short, torch.where(has[..., None], long, short), has

    def forward(s, X, A, W, M):
        N, T, _ = X.shape
        e = s.embed(X, A, M); short, long, has = s.views(e, M)
        if s.ead: mh = torch.sigmoid(s.mal(torch.cat([short, long, short - long], -1))).squeeze(-1) * M
        else: long = short; mh = torch.zeros(N, T, device=X.device)
        x = short; Wx = s.Ws(x)
        l1 = (Wx * s.a1).sum(-1).T; l2 = (Wx * s.a2).sum(-1).T
        logit = F.leaky_relu(l1[:, :, None] + l2[:, None, :], 0.2)
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
        allow = ((di >= 0) & ((di <= s.Da) | (idx[None, :] == 0)))[None, None] & M[:, None, None, :]
        att = torch.softmax(sc.masked_fill(~allow, -1e9), -1) * allow
        z = s.Wo(torch.einsum("nsij,njsd->nisd", att, v).reshape(N, T, -1)) * M[..., None]
        yh = torch.sigmoid(s.head(torch.cat([z, short], -1) if s.skip else z)).squeeze(-1)
        return dict(trust=yh, mal=mh, short=short, long=long, has=has)


def cssl_multi(outs, P, beta=0.1):
    """InfoNCE over all training tracklets (all segments): last-k view vs robust history, other tracklets = negatives."""
    sh, lg = [], []
    for o, s in zip(outs, P):
        M = s["M"]; T = M.shape[1]; last = (M.long() * torch.arange(T)[None]).argmax(1); ar = torch.arange(M.shape[0])
        sel = (s["sp"] == 0) & M.any(1) & o["has"][ar, last]
        if sel.any(): sh.append(o["short"][ar[sel], last[sel]]); lg.append(o["long"][ar[sel], last[sel]])
    if not sh or sum(len(x) for x in sh) < 4: return outs[0]["short"].sum() * 0
    sh = F.normalize(torch.cat(sh), dim=-1); lg = F.normalize(torch.cat(lg), dim=-1)
    return F.cross_entropy(sh @ lg.T / beta, torch.arange(len(sh)))


# ----------------------------------------------------------------------------- training / evaluation
def conf(y, s, thr):
    p = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    ap = average_precision_score(y, s) if y.sum() > 0 else float("nan")
    return int(tp), int(fp), int(tn), int(fn), float(auc), float(ap)


def _masks(P, role): return [s["M"] & (s["sp"] == role)[:, None] for s in P]


def _scores(net, P, masks):
    net.eval()
    with torch.no_grad(): outs = [net(s["X"], s["A"], s["W"], s["M"]) for s in P]
    sc = torch.cat([1 - o["trust"][m] for o, m in zip(outs, masks)]).numpy()
    y = torch.cat([s["y"][m] for s, m in zip(P, masks)]).numpy()
    return sc, y


def run(P, seed, mar=True, cssl=True, ead=True, epochs=250, pre=80, patience=40, lam=(1.0, 0.1, 0.5),
        k=3, d=64, drop=0.1, wd=1e-4, skip=False, eval_every=1):
    torch.manual_seed(seed); np.random.seed(seed)
    net = TVCL(P[0]["X"].shape[-1], d=d, mar=mar, ead=ead, k=k, drop=drop, skip=skip)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=wd)
    mtr, mva, mte = _masks(P, 0), _masks(P, 1), _masks(P, 2)
    ytr = torch.cat([s["y"][m] for s, m in zip(P, mtr)]); ttr = torch.cat([s["trust"][m] for s, m in zip(P, mtr)])
    pos = ytr.mean().clamp(1e-3, 1 - 1e-3); wpos = (1 - pos) / pos
    fwd = lambda: [net(s["X"], s["A"], s["W"], s["M"]) for s in P]
    if cssl and ead:
        net.train()
        for _ in range(pre):                                                      # phase 1: label-free
            opt.zero_grad(); cssl_multi(fwd(), P).backward(); opt.step()
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):                                                      # phase 2: joint fine-tuning
        net.train(); opt.zero_grad(); outs = fwd()
        tr_hat = torch.cat([o["trust"][m] for o, m in zip(outs, mtr)]); mal = torch.cat([o["mal"][m] for o, m in zip(outs, mtr)])
        wt = torch.where(ytr > 0.5, wpos, torch.ones(()))
        l_tr = (wt * (tr_hat - ttr) ** 2).mean()
        l_mal = F.binary_cross_entropy(mal.clamp(1e-6, 1 - 1e-6), ytr, weight=wt) if ead else 0 * l_tr
        l_cs = cssl_multi(outs, P) if (cssl and ead) else 0 * l_tr
        (lam[0] * l_tr + lam[1] * l_cs + lam[2] * l_mal).backward(); opt.step()
        if ep % eval_every: continue
        sv, yv = _scores(net, P, mva)
        a = roc_auc_score(yv, sv) if 0 < yv.sum() < len(yv) else 0.5
        if a > best + 1e-4: best, best_state, bad = a, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state)
    sv, yv = _scores(net, P, mva); fpr, tpr, th = roc_curve(yv, sv); thr = th[np.argmax(tpr - fpr)]   # Youden on val
    st, yt = _scores(net, P, mte)
    return conf(yt, st, thr)


def _flat(P, hist, role):
    xs, ys = [], []
    for s in P:
        X = s["X"].numpy(); M = s["M"].numpy()
        if hist:
            cm = np.cumsum(X * M[..., None], 1); cn = np.cumsum(M, 1)[..., None]
            prev = np.zeros_like(X); prev[:, 1:] = cm[:, :-1] / np.maximum(cn[:, :-1], 1)
            X = np.concatenate([X, X - prev], -1)
        m = M & (s["sp"].numpy() == role)[:, None]
        xs.append(X[m]); ys.append(s["y"].numpy()[m])
    return np.concatenate(xs), np.concatenate(ys)


def baseline(P, kind):
    hist = kind == "GBM+hist"
    (Xt, yt), (Xv, yv), (Xe, ye) = (_flat(P, hist, r) for r in (0, 1, 2))
    mdl = LogisticRegression(max_iter=1000, class_weight="balanced") if kind == "LogReg" else \
        HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0)
    mdl.fit(Xt, yt); sv = mdl.predict_proba(Xv)[:, 1]; fpr, tpr, th = roc_curve(yv, sv)
    return conf(ye, mdl.predict_proba(Xe)[:, 1], th[np.argmax(tpr - fpr)])


def diag(P, k=3):
    net = TVCL(P[0]["X"].shape[-1], k=k); tot = has = nodes = elig = 0
    with torch.no_grad():
        for s in P:
            o = net(s["X"], s["A"], s["W"], s["M"]); M = s["M"]; T = M.shape[1]
            last = (M.long() * torch.arange(T)[None]).argmax(1); ar = torch.arange(M.shape[0])
            tot += int(M.sum()); has += int((o["has"] & M).sum()); nodes += M.shape[0]; elig += int((M.any(1) & o["has"][ar, last]).sum())
    print("segments=%d tracklets=%d node-snapshots=%d (%.2f per tracklet)" % (len(P), nodes, tot, tot / nodes))
    print("node-snapshots with a long-term view: %d (%.1f%%); tracklets eligible for CSSL: %d (%.1f%%)" %
          (has, 100 * has / tot, elig, 100 * elig / nodes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="runs_real.csv"); ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--T", type=int, default=48); ap.add_argument("--time-scale", type=float, default=1e9)
    ap.add_argument("--epochs", type=int, default=250); ap.add_argument("--pre", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=1); ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--no-road", action="store_true"); ap.add_argument("--skip", action="store_true")
    ap.add_argument("--diag-only", action="store_true"); ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()
    df = load_dir(a.data, a.time_scale, a.max_files)
    print("messages=%d senders=%d receivers=%d attacker-senders=%.1f%%  time span=%.0fs" % (
        len(df), df.send.nunique(), df.recv.nunique(), 100 * df.groupby("send").att.max().mean(), df.t.max() - df.t.min()))
    feats = [f for f in FEATS_ALL if not (a.no_road and f == "road")]
    segs = make_graph(df, feats, a.dt, a.T)
    cond = "constantPositionOffset" + ("_noroad" if a.no_road else "")
    new = not os.path.exists(a.out); f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new: w.writerow("table,dataset,condition,model,seed,tp,fp,tn,fn,auc,pr_auc".split(","))
    jobs = [("TVCL (Full)", {}), ("w/o MAR", dict(mar=False)), ("w/o CSSL", dict(cssl=False)), ("w/o EAD", dict(ead=False, cssl=False))]
    for sd in a.seeds:
        P = prepare(segs, df, sd)
        if sd == a.seeds[0]:
            diag(P)
            if a.diag_only: raise SystemExit
        for nm, kw in jobs:
            r = run(P, sd, epochs=a.epochs, pre=a.pre, skip=a.skip, eval_every=a.eval_every, **kw)
            w.writerow(["main", "intas-real", cond, nm, sd, *r]); f.flush()
            print(sd, nm, "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f" % r, flush=True)
        if not a.no_baselines:
            for nm, kd in (("GBM (msg features)", "GBM"), ("LogReg (msg features)", "LogReg"), ("GBM (+own-history)", "GBM+hist")):
                r = baseline(P, kd); w.writerow(["main", "intas-real", cond, nm, sd, *r]); f.flush(); print(sd, nm, "auc=%.3f" % r[4], flush=True)
