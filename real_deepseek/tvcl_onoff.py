#!/usr/bin/env python3
"""
Ablation runner on the REAL InTAS dataset (veh_*.json), with Wilcoxon significance
test across seeds.  No synthetic data.

Usage
-----
python tvcl_onoff.py --data "C:\\Users\\hassan\\TVCL_NEW\\pythonProject1\\dataset" \
       --seeds 0 1 2 3 4 5 6 7 8 9 --out ablation_intas.csv
"""
import argparse, os, csv
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from real_deepseek.OK_MAR.feats import load


def main():
    import tvcl_intas as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="ablation_intas.csv")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()

    df = load(a.data)
    print("[load] messages=%d  vehicles=%d  attack rate=%.3f"
          % (len(df), df.send.nunique(), df.att.mean()))

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
        for nm, kw in jobs:
            r = T.run(D, sd, epochs=a.epochs, **kw)
            w.writerow(["ablation", "intas", "default", nm, sd, *r])
            f.flush()
            print(sd, nm, "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f" % r)
            results.append(dict(model=nm, seed=sd, auc=r[4], ap=r[5]))
    f.close()

    # ---- significance across seeds ----
    if len(a.seeds) >= 5 and not a.no_ablation:
        res = pd.DataFrame(results)
        full = res[res.model == "TVCL (Full)"].sort_values("seed")
        print("\n=== Wilcoxon signed-rank vs. TVCL (Full), n_seeds=%d ==="
              % len(a.seeds))
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