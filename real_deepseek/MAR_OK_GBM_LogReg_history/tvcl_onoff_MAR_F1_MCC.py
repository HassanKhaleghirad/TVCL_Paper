#!/usr/bin/env python3
"""
Ablation runner on real InTAS (veh_*.json).
Reports TVCL (Full) and 4 ablations, AND always three baselines:
  * GBM (msg features)
  * LogReg (msg features)
  * GBM (+own-history features)   <-- matches your earlier log
All five TVCL runs + three baselines share the same vehicle-disjoint split per seed.

Usage:
python tvcl_onoff_MAR_F1_MCC.py --data "C:\\Users\\hassan\\TVCL_NEW\\pythonProject1\\dataset" \
       --seeds 0 1 2 3 4 5 6 7 8 9 --out ablation_intas_final.csv
"""
import argparse, os, csv
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from feats import load


def main():
    import tvcl_intas_new as T   # <-- use the file you actually have

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="ablation_intas.csv")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()

    df = load(a.data)
    print("[load] messages=%d  vehicles=%d  attack rate=%.3f"
          % (len(df), df.send.nunique(), df.att.mean()))

    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow("table,dataset,condition,model,seed,"
                   "tp,fp,tn,fn,auc,pr_auc,f1,mcc".split(","))

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

        # ------- TVCL + ablations -------
        for nm, kw in jobs:
            r = T.run(D, sd, epochs=a.epochs, **{**eval(a.cfg), **kw})
            w.writerow(["ablation", "intas", "default", nm, sd, *r])
            f.flush()
            print(sd, nm,
                  "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f f1=%.3f mcc=%.3f" % r)
            results.append(dict(model=nm, seed=sd,
                                auc=r[4], ap=r[5], f1=r[6], mcc=r[7]))

        # ------- baselines (always reported) -------
        for bname in ("GBM (msg features)",
                      "LogReg (msg features)",
                      "GBM (+own-history features)"):
            r = T.baseline(D, bname)
            w.writerow(["baseline", "intas", "default", bname, sd, *r])
            f.flush()
            print(sd, bname,
                  "tp/fp/tn/fn=%d/%d/%d/%d auc=%.3f ap=%.3f f1=%.3f mcc=%.3f" % r)
            results.append(dict(model=bname, seed=sd,
                                auc=r[4], ap=r[5], f1=r[6], mcc=r[7]))

    f.close()

    # ------- significance across seeds (vs Full) -------
    if len(a.seeds) >= 5 and not a.no_ablation:
        res = pd.DataFrame(results)
        full = res[res.model == "TVCL (Full)"].sort_values("seed")
        print("\n=== Wilcoxon signed-rank vs. TVCL (Full), n_seeds=%d ==="
              % len(a.seeds))
        for m in ["w/o MAR", "w/o CSSL", "w/o EAD", "w/o all temporal",
                  "GBM (msg features)", "LogReg (msg features)",
                  "GBM (+own-history features)"]:
            other = res[res.model == m].sort_values("seed")
            if len(other) != len(full) or len(full) < 5:
                continue
            line = "%-30s" % m
            for metric in ("auc", "ap", "f1", "mcc"):
                try:
                    p = wilcoxon(full[metric].values,
                                 other[metric].values).pvalue
                except ValueError:
                    p = float("nan")
                line += "  Δ%-4s=%+.3f(p=%.3f)" % (
                    metric.upper(),
                    float(np.median(full[metric].values - other[metric].values)),
                    p)
            print(line)


if __name__ == "__main__":
    main()