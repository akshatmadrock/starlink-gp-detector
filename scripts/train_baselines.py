"""
Train and evaluate all baseline detectors.
Run from repo root:  python scripts/train_baselines.py
"""

import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from gp_detector.dataset   import load_raw, engineer_features, make_labeled_split, \
                                   chronological_split, get_arrays, FEATURE_COLS
from gp_detector.models.baselines import ThresholdDetector, build_logreg, build_mlp
from gp_detector.evaluate  import compute_all, stratified

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

print("Loading data...")
df, soft = make_labeled_split(engineer_features(load_raw()))
tr, va, te = chronological_split(df)
X_tr, y_tr = get_arrays(tr)
X_te, y_te = get_arrays(te)
elev_te    = te["sat_elevation"].values
print(f"  Train {len(X_tr)} | Test {len(X_te)} | pos rate {y_tr.mean():.3f}")

all_metrics = {}

# Fixed threshold
thresh = ThresholdDetector(z_col_idx=FEATURE_COLS.index("z_score"))
p_th   = thresh.predict_proba(X_te)[:, 1]
all_metrics["Threshold"] = compute_all(y_te, p_th, "Threshold")
print(f"Threshold   ECE={all_metrics['Threshold']['ECE']:.4f}  "
      f"AUROC={all_metrics['Threshold']['AUROC']:.4f}")

# Logistic Regression
lr = build_logreg(); lr.fit(X_tr, y_tr)
p_lr = lr.predict_proba(X_te)[:, 1]
all_metrics["LogReg"] = compute_all(y_te, p_lr, "LogReg")
pickle.dump(lr, open(RESULTS / "logreg.pkl", "wb"))
print(f"LogReg      ECE={all_metrics['LogReg']['ECE']:.4f}  "
      f"AUROC={all_metrics['LogReg']['AUROC']:.4f}")

# MLP
mlp = build_mlp(); mlp.fit(X_tr, y_tr)
p_mlp = mlp.predict_proba(X_te)[:, 1]
all_metrics["MLP"] = compute_all(y_te, p_mlp, "MLP")
pickle.dump(mlp, open(RESULTS / "mlp.pkl", "wb"))
print(f"MLP         ECE={all_metrics['MLP']['ECE']:.4f}  "
      f"AUROC={all_metrics['MLP']['AUROC']:.4f}")

json.dump(all_metrics, open(RESULTS / "baseline_metrics.json", "w"), indent=2)
print(f"\nSaved to results/")
