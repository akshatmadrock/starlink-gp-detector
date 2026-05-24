"""
Three targeted enhancements over the Phase 3 baseline:

1. Kernel lengthscales — extract learned lengthscales from the trained full
   model and plot which features the GP actually relies on.

2. Angular velocity feature (elev_rate) — retrain with the expanded geometric
   block (elevation + elev_rate) and compare ECE to Phase 3 baseline.

3. Pass-level aggregation — group per-window GP probabilities by satellite
   pass and make a single pass-level detection decision via median threshold.
   Compare precision/recall to per-window threshold detector.

Results saved to results/enhancements.json
Figures saved to  results/figures/

Run from repo root:  python scripts/enhancements.py
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gp_detector.dataset import (
    load_raw, engineer_features, make_labeled_split,
    chronological_split, get_arrays, FEATURE_COLS,
)
from gp_detector.models.svgp import train, predict, GEOM_DIMS, SPEC_DIMS, TIME_DIMS
from gp_detector.evaluate import compute_all

RESULTS = Path("results")
FIGS    = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df, soft_df = make_labeled_split(engineer_features(load_raw()))
tr, va, te  = chronological_split(df)
X_tr, y_tr  = get_arrays(tr)
X_va, y_va  = get_arrays(va)
X_te, y_te  = get_arrays(te)
print(f"  Train {len(X_tr):,} | Val {len(X_va):,} | Test {len(X_te):,}")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 1. Train full model + extract kernel lengthscales ─────────────────────────
print("\n[1] Training full model (elev_rate included)...")
model, lik, scaler, losses = train(
    X_tr, y_tr,
    n_inducing=300,
    n_epochs=80,
    lr=0.05,
    variant="full",
    verbose=False,
)

# Extract learned lengthscales from each kernel block
import torch
kernels = model.covar_module.kernels  # list of ScaleKernel
ls_geom = kernels[0].base_kernel.lengthscale.detach().squeeze().tolist()
ls_spec = kernels[1].base_kernel.lengthscale.detach().squeeze().tolist()
ls_time = kernels[2].base_kernel.lengthscale.detach().squeeze().tolist()

# Map back to feature names
feature_ls = {}
for i, dim in enumerate(GEOM_DIMS):
    val = ls_geom[i] if isinstance(ls_geom, list) else ls_geom
    feature_ls[FEATURE_COLS[dim]] = float(val)
for i, dim in enumerate(SPEC_DIMS):
    val = ls_spec[i] if isinstance(ls_spec, list) else ls_spec
    feature_ls[FEATURE_COLS[dim]] = float(val)
for i, dim in enumerate(TIME_DIMS):
    val = ls_time[i] if isinstance(ls_time, list) else ls_time
    feature_ls[FEATURE_COLS[dim]] = float(val)

print("  Learned lengthscales:")
for k, v in feature_ls.items():
    print(f"    {k:20s}: {v:.4f}")

# Plot lengthscales
fig, ax = plt.subplots(figsize=(7, 4))
names = list(feature_ls.keys())
vals  = list(feature_ls.values())
colors = []
for n in names:
    if n in [FEATURE_COLS[d] for d in GEOM_DIMS]:
        colors.append("#2196F3")   # blue = geom
    elif n in [FEATURE_COLS[d] for d in SPEC_DIMS]:
        colors.append("#FF5722")   # orange = spec
    else:
        colors.append("#4CAF50")   # green = time

bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Learned RBF lengthscale (lower = more informative)")
ax.set_title("GP Kernel Lengthscales by Feature")
ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="ℓ = 1")

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#FF5722", label="Spectral block"),
    Patch(facecolor="#2196F3", label="Geometric block"),
    Patch(facecolor="#4CAF50", label="Temporal block"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(FIGS / "lengthscales.png", dpi=150)
plt.close()
print("  Saved results/figures/lengthscales.png")

# ECE with elev_rate
p_va, _   = predict(model, lik, scaler, X_va)
p_te, _   = predict(model, lik, scaler, X_te)
m_va_new  = compute_all(y_va, p_va)
m_te_new  = compute_all(y_te, p_te)
print(f"  val_ECE={m_va_new['ECE']:.5f}  te_ECE={m_te_new['ECE']:.5f}  AUROC={m_te_new['AUROC']:.4f}")

# ── 2. Compare ECE: baseline (no elev_rate) vs enhanced ───────────────────────
print("\n[2] Baseline ECE (from ablation.json)...")
baseline_ece = json.load(open(RESULTS / "ablation.json"))["full"]["te_ece"]
print(f"  Baseline (Phase 3 full, 8 features): te_ECE = {baseline_ece:.5f}")
print(f"  Enhanced (+ elev_rate, 9 features):  te_ECE = {m_te_new['ECE']:.5f}")
delta = m_te_new["ECE"] - baseline_ece
print(f"  Delta: {delta:+.5f}  ({'improved' if delta < 0 else 'no improvement'})")

# ── 3. Pass-level aggregation ─────────────────────────────────────────────────
print("\n[3] Pass-level aggregation...")

import pandas as pd

# te is already chronological (from chronological_split); reset index so
# positions align with p_te (which was computed from get_arrays(te)).
te_full = te.reset_index(drop=True)
te_full["gp_prob"]       = p_te
te_full["threshold_det"] = (te_full["z_score"] >= 4.0).astype(int)
te_full["label"]         = y_te

# Reconstruct pass IDs within the test split
time_gap           = te_full["unix_time"].diff()
new_pass           = (te_full["sat_name"] != te_full["sat_name"].shift()) | \
                     (time_gap > 30) | time_gap.isna()
te_full["pass_id"] = new_pass.cumsum()

# Pass-level aggregation: a pass is "detected" if median GP prob > 0.5
pass_gp = te_full.groupby("pass_id").agg(
    gp_median=("gp_prob",       "median"),
    gp_max   =("gp_prob",       "max"),
    thr_any  =("threshold_det", "max"),   # threshold: detected if any window fires
    true_pos =("label",         "max"),   # pass is positive if any HARD window present
    n_windows=("label",         "count"),
).reset_index()

pass_gp["gp_det_median"] = (pass_gp["gp_median"] > 0.5).astype(int)
pass_gp["gp_det_max"]    = (pass_gp["gp_max"]    > 0.5).astype(int)

def prf(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "tp": int(tp), "fp": int(fp), "fn": int(fn)}

r_thr    = prf(pass_gp["true_pos"], pass_gp["thr_any"])
r_gp_med = prf(pass_gp["true_pos"], pass_gp["gp_det_median"])
r_gp_max = prf(pass_gp["true_pos"], pass_gp["gp_det_max"])

print(f"  Total test passes:   {len(pass_gp):,}")
print(f"  Positive passes:     {pass_gp['true_pos'].sum():,}")
print(f"  Threshold (any>4.0): P={r_thr['precision']:.3f}  R={r_thr['recall']:.3f}  F1={r_thr['f1']:.3f}  FP={r_thr['fp']}")
print(f"  GP median>0.5:       P={r_gp_med['precision']:.3f}  R={r_gp_med['recall']:.3f}  F1={r_gp_med['f1']:.3f}  FP={r_gp_med['fp']}")
print(f"  GP max>0.5:          P={r_gp_max['precision']:.3f}  R={r_gp_max['recall']:.3f}  F1={r_gp_max['f1']:.3f}  FP={r_gp_max['fp']}")

# Plot pass-level comparison
fig, ax = plt.subplots(figsize=(7, 4))
methods = ["Threshold\n(any>4.0)", "GP median>0.5", "GP max>0.5"]
prec    = [r_thr["precision"], r_gp_med["precision"], r_gp_max["precision"]]
rec     = [r_thr["recall"],    r_gp_med["recall"],    r_gp_max["recall"]]
f1s     = [r_thr["f1"],        r_gp_med["f1"],        r_gp_max["f1"]]
x       = np.arange(3)
w       = 0.25
ax.bar(x - w, prec, w, label="Precision", color="#2196F3")
ax.bar(x,     rec,  w, label="Recall",    color="#FF5722")
ax.bar(x + w, f1s,  w, label="F1",        color="#4CAF50")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Pass-Level Detection: Precision / Recall / F1")
ax.legend(); plt.tight_layout()
fig.savefig(FIGS / "pass_level.png", dpi=150)
plt.close()
print("  Saved results/figures/pass_level.png")

# ── Save results ───────────────────────────────────────────────────────────────
results = {
    "feature_lengthscales": feature_ls,
    "ece_comparison": {
        "baseline_te_ece":  baseline_ece,
        "enhanced_te_ece":  m_te_new["ECE"],
        "enhanced_val_ece": m_va_new["ECE"],
        "enhanced_auroc":   m_te_new["AUROC"],
        "delta":            delta,
    },
    "pass_level": {
        "total_passes":    int(len(pass_gp)),
        "positive_passes": int(pass_gp["true_pos"].sum()),
        "threshold":       r_thr,
        "gp_median":       r_gp_med,
        "gp_max":          r_gp_max,
    },
}
json.dump(results, open(RESULTS / "enhancements.json", "w"), indent=2)
print("\nSaved results/enhancements.json")
print("\nDone.")
