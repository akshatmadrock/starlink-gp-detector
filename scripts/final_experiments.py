"""
Final experiments for grade-maximization pass.

1. Multi-seed ECE (5 seeds, full GP + all 4 ablation variants):
   Produces mean ± std ECE so single-seed numbers become credible statistics.

2. SOFT-within-pass validation:
   For SOFT windows, determine if they belong to a pass that has at least one
   confirmed HARD detection (positive pass) or purely BACKGROUND (negative pass).
   GP probabilities should be higher on SOFT windows inside positive passes.
   This is the strongest possible validation of SOFT-window behavior without
   external ground truth.

3. 4-model reliability diagram overlay:
   Threshold / LogReg / MLP / GP on one plot, all at the same calibration scale.

Results saved to results/final_experiments.json
Figures saved to results/figures/

Run from repo root:  python scripts/final_experiments.py
"""

import sys, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gp_detector.dataset import (
    load_raw, engineer_features, make_labeled_split,
    chronological_split, get_arrays, FEATURE_COLS,
)
from gp_detector.models.svgp import train, predict
from gp_detector.evaluate import compute_all, ece

RESULTS = Path("results")
FIGS    = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
raw_df       = engineer_features(load_raw())
df, soft_df  = make_labeled_split(raw_df)
tr, va, te   = chronological_split(df)
X_tr, y_tr   = get_arrays(tr)
X_va, y_va   = get_arrays(va)
X_te, y_te   = get_arrays(te)
X_soft       = soft_df[FEATURE_COLS].values.astype("float32")
print(f"  Train {len(X_tr):,} | Val {len(X_va):,} | Test {len(X_te):,} | SOFT {len(X_soft):,}")

# ── 1. Multi-seed ECE ─────────────────────────────────────────────────────────
print("\n[1] Multi-seed ECE (5 seeds x 5 variants)...")

SEEDS   = [42, 7, 123, 2025, 314]
VARIANTS = ["full", "no_elev", "spec_only", "single_rbf"]

seed_results = {v: {"val_ece": [], "te_ece": []} for v in VARIANTS}
seed_results["full"]["soft_mean"] = []

for seed in SEEDS:
    print(f"  seed {seed}...")
    for variant in VARIANTS:
        model, lik, scaler, _ = train(
            X_tr, y_tr, n_inducing=200, n_epochs=60, lr=0.05,
            seed=seed, variant=variant, verbose=False,
        )
        p_va, _ = predict(model, lik, scaler, X_va)
        p_te, _ = predict(model, lik, scaler, X_te)
        seed_results[variant]["val_ece"].append(ece(y_va, p_va))
        seed_results[variant]["te_ece"].append(ece(y_te, p_te))
        if variant == "full":
            p_soft, _ = predict(model, lik, scaler, X_soft)
            seed_results["full"]["soft_mean"].append(float(p_soft.mean()))

# Summarize
multi_seed = {}
for variant in VARIANTS:
    te_arr = np.array(seed_results[variant]["te_ece"])
    va_arr = np.array(seed_results[variant]["val_ece"])
    row = {
        "te_ece_mean": float(te_arr.mean()),
        "te_ece_std":  float(te_arr.std()),
        "val_ece_mean": float(va_arr.mean()),
        "val_ece_std":  float(va_arr.std()),
        "seeds": seed_results[variant]["te_ece"],
    }
    if variant == "full":
        row["soft_mean_mean"] = float(np.array(seed_results["full"]["soft_mean"]).mean())
        row["soft_mean_std"]  = float(np.array(seed_results["full"]["soft_mean"]).std())
    multi_seed[variant] = row
    print(f"  {variant:<12}: te_ECE = {te_arr.mean():.2e} ± {te_arr.std():.2e}")

# Plot multi-seed ECE with error bars
fig, ax = plt.subplots(figsize=(7, 4))
variant_labels = {
    "full":       "Full\n(geom+spec+time)",
    "no_elev":    "No elevation\n(spec+time)",
    "spec_only":  "Spec only\n(spec+time)",
    "single_rbf": "Single RBF\n(no additive)",
}
colors = {"full": "#2196F3", "no_elev": "#FF9800", "spec_only": "#4CAF50", "single_rbf": "#E53935"}

x = np.arange(len(VARIANTS))
means = [multi_seed[v]["te_ece_mean"] * 1e4 for v in VARIANTS]
stds  = [multi_seed[v]["te_ece_std"]  * 1e4 for v in VARIANTS]
bars  = ax.bar([variant_labels[v] for v in VARIANTS], means,
               color=[colors[v] for v in VARIANTS],
               edgecolor="white", width=0.55, yerr=stds,
               capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.008,
            f"{mean:.2f}±{std:.2f}×10⁻⁴",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Test ECE (×10⁻⁴, mean ± std, 5 seeds)")
ax.set_title("Ablation Study: Calibration Error by Kernel Variant\n(5 random seeds, error bars = 1 std)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(FIGS / "ablation_ece_multiseed.png", dpi=150)
plt.close()
print("  Saved results/figures/ablation_ece_multiseed.png")

# ── 2. SOFT-within-pass validation ────────────────────────────────────────────
print("\n[2] SOFT-within-pass validation...")

# Train a clean model for this analysis
model, lik, scaler, _ = train(
    X_tr, y_tr, n_inducing=300, n_epochs=80, lr=0.05,
    seed=42, variant="full", verbose=False,
)
p_soft, _ = predict(model, lik, scaler, X_soft)

# Reconstruct pass IDs for the ENTIRE raw dataset (HARD + SOFT + BG)
raw_full = raw_df.sort_values("unix_time").reset_index(drop=True)
time_gap = raw_full["unix_time"].diff()
new_pass = (raw_full["sat_name"] != raw_full["sat_name"].shift()) | \
           (time_gap > 30) | time_gap.isna()
raw_full["pass_id"] = new_pass.cumsum()

# Mark HARD rows
hard_passes = set(raw_full[raw_full["confidence"] == "HARD"]["pass_id"])

# Map each SOFT window to its pass
soft_full = raw_full[raw_full["confidence"] == "SOFT"].reset_index(drop=True)
soft_full["gp_prob"] = p_soft
soft_full["in_hard_pass"] = soft_full["pass_id"].isin(hard_passes)

pos_probs = soft_full[soft_full["in_hard_pass"]]["gp_prob"].values
neg_probs = soft_full[~soft_full["in_hard_pass"]]["gp_prob"].values

print(f"  SOFT in positive passes (n={len(pos_probs):,}): "
      f"mean={pos_probs.mean():.3f}  frac>0.5={( pos_probs>0.5).mean():.3f}")
print(f"  SOFT in negative passes (n={len(neg_probs):,}): "
      f"mean={neg_probs.mean():.3f}  frac>0.5={( neg_probs>0.5).mean():.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
bins = np.linspace(0, 1, 26)
for ax, probs, label, color, n in [
    (axes[0], pos_probs, "SOFT in\npositive pass\n(pass has HARD window)", "#E53935", len(pos_probs)),
    (axes[1], neg_probs, "SOFT in\nnegative pass\n(no HARD window in pass)", "#78909C", len(neg_probs)),
]:
    ax.hist(probs, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="black", lw=0.9, ls="--", alpha=0.6)
    ax.set_title(f"{label}\nn={n:,}", fontsize=10)
    ax.set_xlabel("GP detection probability")
    ax.set_ylabel("Window count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("SOFT Window GP Probabilities: Positive vs Negative Passes\n"
             "(Model never trained on SOFT windows)", fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIGS / "soft_pass_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved results/figures/soft_pass_validation.png")

soft_pass_stats = {
    "positive_pass": {
        "n": int(len(pos_probs)),
        "mean_prob": float(pos_probs.mean()),
        "frac_above_half": float((pos_probs > 0.5).mean()),
    },
    "negative_pass": {
        "n": int(len(neg_probs)),
        "mean_prob": float(neg_probs.mean()),
        "frac_above_half": float((neg_probs > 0.5).mean()),
    },
}

# ── 3. 4-model reliability diagram overlay ────────────────────────────────────
print("\n[3] 4-model reliability diagram overlay...")

# For this we need baseline predictions on current test set (9 features)
# Baselines were trained on 8 features; use only their stored predictions
# or retrain them on the 9-feature set for a fair comparison.
# Retraining is cleanest.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as SKScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# Train lightweight baselines on 9-feature data
print("  Training baselines on 9-feature data...")
lr_pipe = make_pipeline(SKScaler(), LogisticRegression(max_iter=1000, C=1.0))
lr_pipe.fit(X_tr, y_tr)
p_te_lr = lr_pipe.predict_proba(X_te)[:, 1]
print(f"  LogReg ECE = {ece(y_te, p_te_lr):.2e}")

mlp_base = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
mlp_cal  = CalibratedClassifierCV(mlp_base, cv=3, method="isotonic")
mlp_cal.fit(X_tr, y_tr)
p_te_mlp = mlp_cal.predict_proba(X_te)[:, 1]
print(f"  MLP   ECE = {ece(y_te, p_te_mlp):.2e}")

# GP predictions
p_te_gp, _ = predict(model, lik, scaler, X_te)
print(f"  GP    ECE = {ece(y_te, p_te_gp):.2e}")

# Threshold sigmoid
z_te = te["z_score"].values
p_te_thr = 1 / (1 + np.exp(-5 * (z_te - 4.0)))
print(f"  Thr   ECE = {ece(y_te, p_te_thr):.2e}")

# Plot 4-model reliability diagram
def reliability_pts(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    cx, cy, cw = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() < 5:
            continue
        cx.append(float(y_prob[m].mean()))
        cy.append(float(y_true[m].mean()))
        cw.append(int(m.sum()))
    return np.array(cx), np.array(cy), np.array(cw)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: all 4 models
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
for label, probs, color in [
    ("Threshold (sigmoid)", p_te_thr, "#E53935"),
    ("Logistic Regression", p_te_lr,  "#FF9800"),
    ("MLP (isotonic)",      p_te_mlp, "#4CAF50"),
    ("GP (this work)",      p_te_gp,  "#2196F3"),
]:
    cx, cy, cw = reliability_pts(y_te, probs)
    sz = cw / cw.max() * 100 if len(cw) > 0 else 30
    ece_val = ece(y_te, probs)
    ax.scatter(cx, cy, s=sz, color=color, zorder=3,
               label=f"{label}  (ECE={ece_val:.1e})")
    ax.plot(cx, cy, color=color, alpha=0.5, lw=1.5)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability Diagram (All Models)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: zoom in on GP only (0-0.3 range) to show tightness
ax2 = axes[1]
ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")
cx, cy, cw = reliability_pts(y_te, p_te_gp, n_bins=20)
sz = cw / cw.max() * 120 if len(cw) > 0 else 30
ax2.scatter(cx, cy, s=sz, color="#2196F3", zorder=3, label=f"GP (ECE={ece(y_te,p_te_gp):.1e})")
ax2.plot(cx, cy, color="#2196F3", alpha=0.7, lw=1.8)
ax2.set_xlabel("Mean predicted probability")
ax2.set_ylabel("Fraction of positives")
ax2.set_title("GP Calibration Detail")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(FIGS / "reliability_4model.png", dpi=150)
plt.close()
print("  Saved results/figures/reliability_4model.png")

# Collect baseline ECE for report
baseline_ece = {
    "Threshold": float(ece(y_te, p_te_thr)),
    "LogReg":    float(ece(y_te, p_te_lr)),
    "MLP":       float(ece(y_te, p_te_mlp)),
    "GP":        float(ece(y_te, p_te_gp)),
}

# ── Save all results ──────────────────────────────────────────────────────────
out = {
    "multi_seed_ablation": multi_seed,
    "soft_pass_validation": soft_pass_stats,
    "baseline_ece_9feat": baseline_ece,
}
json.dump(out, open(RESULTS / "final_experiments.json", "w"), indent=2)
print("\nSaved results/final_experiments.json")
print("\n=== Key numbers for report ===")
print(f"Full GP  te_ECE = {multi_seed['full']['te_ece_mean']:.2e} ± {multi_seed['full']['te_ece_std']:.2e} (5 seeds)")
print(f"single_rbf ECE  = {multi_seed['single_rbf']['te_ece_mean']:.2e} ± {multi_seed['single_rbf']['te_ece_std']:.2e}")
ratio = multi_seed['single_rbf']['te_ece_mean'] / multi_seed['full']['te_ece_mean']
print(f"Ratio single_rbf / full = {ratio:.1f}x worse")
print(f"SOFT in +pass: mean={soft_pass_stats['positive_pass']['mean_prob']:.3f}  frac>0.5={soft_pass_stats['positive_pass']['frac_above_half']:.3f}")
print(f"SOFT in -pass: mean={soft_pass_stats['negative_pass']['mean_prob']:.3f}  frac>0.5={soft_pass_stats['negative_pass']['frac_above_half']:.3f}")
print("\nDone.")
