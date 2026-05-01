"""
Train the SVGP detector.
Run from repo root:  python scripts/train_svgp.py
"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from gp_detector.dataset        import load_raw, engineer_features, make_labeled_split, \
                                        chronological_split, get_arrays
from gp_detector.models.svgp    import train, predict, save
from gp_detector.evaluate       import compute_all, stratified
import numpy as np

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

print("Loading data...")
df, _ = make_labeled_split(engineer_features(load_raw()))
tr, va, te = chronological_split(df)
X_tr, y_tr = get_arrays(tr)
X_te, y_te = get_arrays(te)
elev_te    = te["sat_elevation"].values

print(f"  Train {len(X_tr)} | Test {len(X_te)}")
print("Training SVGP...")
t0 = time.time()
model, lik, scaler, losses = train(X_tr, y_tr, n_inducing=300, n_epochs=80, verbose=True)
print(f"  Done in {time.time()-t0:.1f}s")

prob, std = predict(model, lik, scaler, X_te)
metrics   = compute_all(y_te, prob, "GP")
strat     = stratified(y_te, prob, elev_te)
print(f"GP  ECE={metrics['ECE']:.4f}  AUROC={metrics['AUROC']:.4f}")

save(model, lik, scaler, str(RESULTS / "svgp"))
np.save(RESULTS / "svgp_test_preds.npy",
        {"prob": prob, "std": std, "y_te": y_te, "elev_te": elev_te, "losses": losses})
json.dump(metrics, open(RESULTS / "svgp_metrics.json", "w"), indent=2)
json.dump(strat,   open(RESULTS / "svgp_stratified.json", "w"), indent=2)
print("Saved to results/")
