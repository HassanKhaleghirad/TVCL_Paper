# Config selection on VALIDATION AUC only (test vehicles fixed & never used)
import sys, numpy as np
from feats import load
import tvcl_intas as T
df = load(); cfgs = {"base": {}, "skip": dict(skip=True), "skip_d32_drop.3": dict(skip=True, d=32, drop=0.3, wd=1e-3),
        "d32_drop.3": dict(d=32, drop=0.3, wd=1e-3), "skip_lam.5": dict(skip=True, lam=(1.0, 0.5, 0.5))}
Ds = {sd: T.build(df, sd, fixed_test=True) for sd in range(100, 108)}
for n, c in cfgs.items():
    v = [T.run(Ds[sd], sd, epochs=200, ret_val=True, **c)[1] for sd in Ds]
    print(n, "val AUC mean=%.3f std=%.3f" % (np.mean(v), np.std(v)), flush=True)
