# TVCL on InTAS (constantPositionOffset) - vehicle-disjoint protocol

pip install torch scikit-learn pandas numpy scipy
# all veh_*.json files in one folder (default /mnt/user-data/uploads; change with --data)

# 1) config selection on VALIDATION AUC only (test vehicles fixed, never used)
python tune.py
# 2) final run: fixed test vehicles, 20 seeds (inner train/val split + init vary)
python tvcl_intas.py --fixed-test --cfg "dict(skip=True,d=32,drop=0.3,wd=1e-3)" --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --epochs 200 --out detection_runs_intas.csv
# 3) ablation rows (Full / w/o MAR / w/o CSSL / w/o EAD)
python tvcl_intas.py --fixed-test --cfg "dict(skip=True,d=32,drop=0.3,wd=1e-3)" --ablation --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 200 --out ablation_runs_intas.csv

Output CSV = detection_runs.csv format for gen_tables.py (dataset key: intas).
Also written for the same split: GBM and LogReg on the same snapshot features (reference baselines).
