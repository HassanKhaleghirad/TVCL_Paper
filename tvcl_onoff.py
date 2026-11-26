#!/usr/bin/env python3
"""
TVCL on long-lived vehicles with ON-OFF / PATIENT attackers (synthetic, BSM-like).
Reuses build()/run()/conf()/baseline() from tvcl_intas.py unchanged (same vehicle-disjoint protocol).

Why: on InTAS a sender lives ~1.6 snapshots, so the k=3 short/long views never exist (CSSL loss = 0).
Here every vehicle lives 400-1000 s => 4-10 snapshots of 12, so EAD/CSSL/temporal attention are truly active.

Design (disclose in the paper as a controlled SYNTHETIC study, not as real-data evidence):
  * every vehicle has its own noise scale and bias (so a vehicle's own history is informative);
  * attackers inflate position/speed residuals by 2-3 sigma of THEIR OWN noise, only while "on"
    (onoff: periodic duty cycle; patient: benign first 50-70% of life, then persistently on);
  * honest vehicles have random glitches (3% of messages) so single messages are not trivially separable;
  * only BSM-observable quantities are used (no simulator metadata such as distance-to-road-edge).

Usage:
  python tvcl_onoff.py --mode onoff   --seeds 0 1 2 3 4 --out runs_onoff.csv
  python tvcl_onoff.py --mode patient --seeds 0 1 2 3 4 --out runs_patient.csv
  python tvcl_onoff.py --dataset intas --data /path/to/dataset --diag-only      # prints how many (v,t) have a long view
"""
import argparse, os, csv, warnings
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore")

SIM_FEATS = ["pos_res", "spd_res", "acc", "hed_diff", "delay", "dist"]


def simulate(n_veh=240, horizon=1200.0, dt=5.0, frac_att=0.2, mode="onoff", n_reg=12, seed=0):
    rng = np.random.default_rng(seed)
    is_att = np.zeros(n_veh, bool); is_att[rng.choice(n_veh, int(n_veh * frac_att), replace=False)] = True
    parts = []
    for v in range(n_veh):
        L = rng.uniform(400, 1000); t0 = rng.uniform(0, horizon - L)
        ts = np.arange(t0, t0 + L, dt); n = len(ts)
        base = rng.lognormal(0, 0.4); bias = rng.normal(0, 0.3)
        on = np.zeros(n, bool); shift = 0.0
        if is_att[v]:
            shift = base * rng.uniform(2.0, 3.0)
            if mode == "onoff":
                P = rng.uniform(150, 400); ph = rng.uniform(0, P); duty = rng.uniform(0.35, 0.65)
                on = (((ts - t0 + ph) % P) / P) < duty
            else:                                              # patient
                on = ts > t0 + L * rng.uniform(0.5, 0.7)
        glitch = rng.random(n) < 0.03
        amp = on * shift * (1 + 0.3 * rng.standard_normal(n)) + glitch * base * rng.uniform(2, 3, n)
        sgn = rng.choice([-1.0, 1.0], n)
        reg = (np.cumsum(rng.choice([-1, 0, 0, 1], n)) + rng.integers(n_reg)) % n_reg
        parts.append(pd.DataFrame(dict(
            send=v, recv=reg, t=ts, att=on.astype(int),
            pos_res=np.abs(bias + base * rng.standard_normal(n) + sgn * amp),
            spd_res=np.abs(0.4 * base * rng.standard_normal(n) + 0.3 * sgn * amp),
            acc=rng.standard_normal(n) * base, hed_diff=rng.standard_normal(n) * 0.5,
            delay=rng.lognormal(-2, 0.5, n), dist=rng.uniform(0, 300, n))))
    return pd.concat(parts).sort_values("t", kind="stable").reset_index(drop=True)


def baseline_hist(D, conf):
    """Fair history-aware baseline: GBM on [x_t, x_t - causal mean of own past snapshots]."""
    X, M, y, sp = D["X"].numpy(), D["M"].numpy(), D["y"].numpy(), D["split"].numpy()
    cm = np.cumsum(X * M[..., None], 1); cn = np.cumsum(M, 1)[..., None]
    prev = np.zeros_like(X); prev[:, 1:] = cm[:, :-1] / np.maximum(cn[:, :-1], 1)
    Fx = np.concatenate([X, X - prev], -1)
    m = {i: M & (sp == i)[:, None] for i in range(3)}
    mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0).fit(Fx[m[0]], y[m[0]])
    sv = mdl.predict_proba(Fx[m[1]])[:, 1]; fpr, tpr, th = roc_curve(y[m[1]], sv)
    return conf(y[m[2]], mdl.predict_proba(Fx[m[2]])[:, 1], th[np.argmax(tpr - fpr)])


def diag(D, T):
    """How many (vehicle, snapshot) pairs actually have a long-term view? (0 => CSSL/EAD inactive)"""
    M = D["M"]; net = T.TVCL(D["X"].shape[-1], k=3)
    with torch.no_grad():
        o = net(D["X"], D["A"], D["W"], M)
    per = M.sum(1).float()
    print("snapshots/vehicle: mean=%.2f median=%.0f  vehicles with >=4 snapshots: %.1f%%" %
          (per.mean(), per.median(), 100 * (per >= 4).float().mean()))
    print("(v,t) pairs: %d, with long view (has=True): %d (%.1f%%)" %
          (M.sum(), (o["has"] & M).sum(), 100 * (o["has"] & M).sum() / M.sum()))


def main():
    import tvcl_intas as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["sim", "intas"]); ap.add_argument("--data", default="dataset1")
    ap.add_argument("--mode", default="onoff", choices=["onoff", "patient"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--out", default="runs_sim.csv"); ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--skip", action="store_true", help="skip-connection (paper IV-G default is OFF)")
    ap.add_argument("--diag-only", action="store_true"); ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    if a.dataset == "intas":
        from feats import load
        df = load(a.data); name, cond = "intas", "constantPositionOffset"
    else:
        df = simulate(mode=a.mode); T.FEATS = SIM_FEATS; name, cond = "sim", a.mode
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new: w.writerow("table,dataset,condition,model,seed,tp,fp,tn,fn,auc,pr_auc".split(","))
    jobs = [("TVCL (Full)", {})] if a.no_ablation else \
        [("TVCL (Full)", {}), ("w/o MAR", dict(mar=False)), ("w/o CSSL", dict(cssl=False)), ("w/o EAD", dict(ead=False, cssl=False))]
    for sd in a.seeds:
        D = T.build(df, sd)
        if sd == a.seeds[0]: diag(D, T)
        if a.diag_only: return
        for nm, kw in jobs:
            r = T.run(D, sd, epochs=a.epochs, skip=a.skip, **kw)
            w.writerow(["ablation", name, cond, nm, sd, *r]); f.flush()
            print(sd, nm, "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f" % r)
        for nm, fn in (("GBM (msg features)", lambda: T.baseline(D, "GBM")),
                       ("LogReg (msg features)", lambda: T.baseline(D, "LogReg")),
                       ("GBM (+own-history features)", lambda: baseline_hist(D, T.conf))):
            r = fn(); w.writerow(["ablation", name, cond, nm, sd, *r]); f.flush(); print(sd, nm, "auc=%.3f" % r[4])


if __name__ == "__main__":
    main()