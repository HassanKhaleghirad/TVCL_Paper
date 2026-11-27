#!/usr/bin/env python3
"""
TVCL on long-lived vehicles with ON-OFF / PATIENT attackers (synthetic, BSM-like).
Uses build()/run()/conf()/baseline() from tvcl_intas.py (same vehicle-disjoint protocol).

This script is designed so that CSSL / EAD / MAR actually matter:
  * every vehicle lives 400-1000 s  =>  4-10 snapshots (k=3 views exist);
  * attackers stay "on" for >= min_on_snaps consecutive snapshots (EAD-friendly);
  * honest glitches last a single snapshot (so single messages are not separable);
  * only BSM-observable quantities are used.

For InTAS, use --rebin K to keep only vehicles with >= K messages and re-index time so
that each message becomes one snapshot.  This is the ONLY way to make the temporal
modules active on InTAS; without it, mean snapshots/vehicle is ~1.1 and CSSL/EAD see
no signal (this is a data limitation, not a model limitation).

Usage
-----
python tvcl_onoff.py --mode onoff   --seeds 0 1 2 3 4 --out runs_onoff.csv
python tvcl_onoff.py --mode patient --seeds 0 1 2 3 4 --out runs_patient.csv
python tvcl_onoff.py --dataset intas --data /path/to/dataset --rebin 4 --diag-only
python tvcl_onoff.py --dataset intas --data /path/to/dataset --rebin 4 \
       --seeds 0 1 2 3 4 5 6 7 8 9 --out ablation_intas_rebin.csv
"""
import argparse, os, csv, warnings
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")

SIM_FEATS = ["pos_res", "spd_res", "acc", "hed_diff", "delay", "dist"]


# ---------------------------------------------------------------- simulation
def _enforce_min_run(mask, min_run):
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i < min_run:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def _enforce_max_run(mask, max_run):
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i > max_run:
                m[i + max_run:j] = False
            i = j
        else:
            i += 1
    return m


def simulate(n_veh=240, horizon=1200.0, dt=5.0, frac_att=0.2,
             mode="onoff", n_reg=12, seed=0, min_on_snaps=4):
    rng = np.random.default_rng(seed)
    is_att = np.zeros(n_veh, bool)
    is_att[rng.choice(n_veh, int(n_veh * frac_att), replace=False)] = True
    parts = []
    for v in range(n_veh):
        L = rng.uniform(400, 1000)
        t0 = rng.uniform(0, horizon - L)
        ts = np.arange(t0, t0 + L, dt)
        n = len(ts)
        base = rng.lognormal(0, 0.4)
        bias = rng.normal(0, 0.3)
        on = np.zeros(n, bool)
        shift = 0.0
        if is_att[v]:
            shift = base * rng.uniform(2.0, 3.0)
            if mode == "onoff":
                P = rng.uniform(150, 400)
                ph = rng.uniform(0, P)
                duty = rng.uniform(0.35, 0.65)
                on_raw = (((ts - t0 + ph) % P) / P) < duty
                on = _enforce_min_run(on_raw, min_on_snaps)
            else:  # patient
                start = t0 + L * rng.uniform(0.5, 0.7)
                on_raw = ts > start
                on = _enforce_min_run(on_raw, min_on_snaps)

        glitch = rng.random(n) < 0.03
        glitch = _enforce_max_run(glitch, max_run=1)

        amp = (on * shift * (1 + 0.3 * rng.standard_normal(n))
               + glitch * base * rng.uniform(2, 3, n))
        sgn = rng.choice([-1.0, 1.0], n)
        reg = (np.cumsum(rng.choice([-1, 0, 0, 1], n)) + rng.integers(n_reg)) % n_reg
        parts.append(pd.DataFrame(dict(
            send=v, recv=reg, t=ts, att=on.astype(int),
            pos_res=np.abs(bias + base * rng.standard_normal(n) + sgn * amp),
            spd_res=np.abs(0.4 * base * rng.standard_normal(n) + 0.3 * sgn * amp),
            acc=rng.standard_normal(n) * base,
            hed_diff=rng.standard_normal(n) * 0.5,
            delay=rng.lognormal(-2, 0.5, n),
            dist=rng.uniform(0, 300, n),
        )))
    return pd.concat(parts).sort_values("t", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- InTAS rebinning
def rebin_intas(df, window=4):
    """Keep only vehicles with >= window messages; re-index t as message index."""
    if df is None or len(df) == 0:
        return df
    df = df.sort_values(["send", "t"]).copy()
    counts = df.groupby("send").size()
    keep = counts[counts >= window].index
    df = df[df["send"].isin(keep)].copy()
    df["t"] = df.groupby("send").cumcount().astype(float)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- history baseline
def baseline_hist(D, conf):
    X, M, y, sp = D["X"].numpy(), D["M"].numpy(), D["y"].numpy(), D["split"].numpy()
    cm = np.cumsum(X * M[..., None], 1)
    cn = np.cumsum(M, 1)[..., None]
    prev = np.zeros_like(X)
    prev[:, 1:] = cm[:, :-1] / np.maximum(cn[:, :-1], 1)
    Fx = np.concatenate([X, X - prev], -1)
    m = {i: M & (sp == i)[:, None] for i in range(3)}
    mdl = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, random_state=0
    ).fit(Fx[m[0]], y[m[0]])
    sv = mdl.predict_proba(Fx[m[1]])[:, 1]
    fpr, tpr, th = roc_curve(y[m[1]], sv)
    return conf(y[m[2]], mdl.predict_proba(Fx[m[2]])[:, 1],
                th[np.argmax(tpr - fpr)])


# ---------------------------------------------------------------- diagnostics
def diag(D, T):
    M = D["M"]
    net = T.TVCL(D["X"].shape[-1], k=3)
    with torch.no_grad():
        o = net(D["X"], D["A"], D["W"], M)
    per = M.sum(1).float()
    n_pairs = int(M.sum())
    n_long = int((o["has"] & M).sum())
    print("snapshots/vehicle: mean=%.2f median=%.0f  vehicles with >=4 snapshots: %.1f%%"
          % (per.mean(), per.median(), 100 * (per >= 4).float().mean()))
    print("(v,t) pairs: %d, with long view (has=True): %d (%.1f%%)"
          % (n_pairs, n_long, 100 * n_long / max(n_pairs, 1)))
    if n_long == 0:
        print("WARNING: no (v,t) pair has has=True.  CSSL/EAD receive NO signal.")
        print("         Use --rebin K for InTAS, or --dataset sim.")


# ---------------------------------------------------------------- main
def main():
    import tvcl_intas as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["sim", "intas"])
    ap.add_argument("--data", default="dataset1")
    ap.add_argument("--rebin", type=int, default=0,
                    help="InTAS: keep vehicles with >= this many messages; re-index t.")
    ap.add_argument("--mode", default="onoff", choices=["onoff", "patient"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--out", default="runs_sim.csv")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--skip", action="store_true")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()

    # -------- data --------
    if a.dataset == "intas":
        from feats import load
        df = load(a.data)
        n_before = df["send"].nunique() if df is not None and len(df) else 0
        if a.rebin > 0:
            df = rebin_intas(df, window=a.rebin)
            n_after = df["send"].nunique() if df is not None and len(df) else 0
            print("[rebin] vehicles: %d -> %d, rows: %d (window=%d)"
                  % (n_before, n_after, 0 if df is None else len(df), a.rebin))
        name, cond = "intas", "constantPositionOffset"
    else:
        df = simulate(mode=a.mode)
        T.FEATS = SIM_FEATS
        name, cond = "sim", a.mode

    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow("table,dataset,condition,model,seed,tp,fp,tn,fn,auc,pr_auc".split(","))

    if a.no_ablation:
        jobs = [("TVCL (Full)", {})]
    else:
        jobs = [
            ("TVCL (Full)",      {}),
            ("w/o MAR",          dict(mar=False)),
            ("w/o CSSL",         dict(cssl=False)),
            ("w/o EAD",          dict(ead=False)),
            ("w/o all temporal", dict(mar=False, cssl=False, ead=False)),
        ]

    results = []

    for sd in a.seeds:
        D = T.build(df, sd, min_snaps_assert=(sd == a.seeds[0]))
        if sd == a.seeds[0]:
            diag(D, T)
            if a.diag_only:
                return
        for nm, kw in jobs:
            r = T.run(D, sd, epochs=a.epochs, skip=a.skip, **kw)
            w.writerow(["ablation", name, cond, nm, sd, *r])
            f.flush()
            print(sd, nm, "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f" % r)
            results.append(dict(model=nm, seed=sd, auc=r[4], ap=r[5]))

        for nm, fn in (
            ("GBM (msg features)",          lambda: T.baseline(D, "GBM")),
            ("LogReg (msg features)",       lambda: T.baseline(D, "LogReg")),
            ("GBM (+own-history features)", lambda: baseline_hist(D, T.conf)),
        ):
            r = fn()
            w.writerow(["ablation", name, cond, nm, sd, *r])
            f.flush()
            print(sd, nm, "auc=%.3f" % r[4])
            results.append(dict(model=nm, seed=sd, auc=r[4], ap=r[5]))

    f.close()

    # -------- Wilcoxon across seeds --------
    if len(a.seeds) >= 5 and not a.no_ablation:
        res = pd.DataFrame(results)
        full = res[res.model == "TVCL (Full)"].sort_values("seed")
        print("\n=== Wilcoxon signed-rank vs. TVCL (Full), n_seeds=%d ===" % len(a.seeds))
        for m in ["w/o MAR", "w/o CSSL", "w/o EAD", "w/o all temporal"]:
            other = res[res.model == m].sort_values("seed")
            if len(other) != len(full) or len(full) < 5:
                continue
            try:
                p = wilcoxon(full.auc.values, other.auc.values).pvalue
            except ValueError:
                p = float("nan")
            print("%-20s  median ΔAUC=%+.3f  median ΔAP=%+.3f  p(AUC)=%.4f"
                  % (m,
                     float(np.median(full.auc.values - other.auc.values)),
                     float(np.median(full.ap.values - other.ap.values)),
                     p))


if __name__ == "__main__":
    main()