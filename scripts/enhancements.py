"""
Three targeted enhancements over the Phase 3 baseline.

1. ARD kernel lengthscales — per-feature lengthscale from additive GP blocks,
   showing which features the model actually finds informative.

2. Angular velocity (elev_rate) feature — retrain with expanded geometric
   block; compare ECE to Phase 3 baseline.

3. Pass-level aggregation — group per-window GP probabilities by satellite
   pass, decide via median/max; compare P/R/F1 to per-window threshold.

Bonus figures generated for report:
  - SOFT probability histogram (HARD / BG / SOFT window GP probs)
  - Reliability diagram (GP vs threshold)
  - Ablation ECE bar chart
  - System architecture flowchart

Run from repo root:  python scripts/enhancements.py
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import pandas as pd

from gp_detector.dataset import (
    load_raw, engineer_features, make_labeled_split,
    chronological_split, get_arrays, FEATURE_COLS,
)
from gp_detector.models.svgp import train, predict, GEOM_DIMS, SPEC_DIMS, TIME_DIMS
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
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 1. Train full model with ARD + elev_rate ───────────────────────────────────
print("\n[1] Training full model (ARD + elev_rate)...")
model, lik, scaler, losses = train(
    X_tr, y_tr,
    n_inducing=300,
    n_epochs=80,
    lr=0.05,
    variant="full",
    verbose=False,
)

# Extract per-feature ARD lengthscales from each kernel block
kernels   = model.covar_module.kernels  # [ScaleKernel_geom, ScaleKernel_spec, ScaleKernel_time]
block_dims = [GEOM_DIMS, SPEC_DIMS, TIME_DIMS]
block_names = ["Geometric", "Spectral", "Temporal"]

feature_ls = {}
for dims, sk in zip(block_dims, kernels):
    ls = sk.base_kernel.lengthscale.detach().squeeze()
    ls_list = ls.tolist() if ls.dim() > 0 else [ls.item()]
    for i, dim in enumerate(dims):
        feature_ls[FEATURE_COLS[dim]] = ls_list[i] if i < len(ls_list) else ls_list[0]

print("  Learned ARD lengthscales:")
for feat, ls_val in feature_ls.items():
    print(f"    {feat:22s}: {ls_val:.4f}")

# --- Plot lengthscales --------------------------------------------------------
FEAT_COLORS = {}
for d in GEOM_DIMS: FEAT_COLORS[FEATURE_COLS[d]] = "#2196F3"
for d in SPEC_DIMS: FEAT_COLORS[FEATURE_COLS[d]] = "#FF5722"
for d in TIME_DIMS: FEAT_COLORS[FEATURE_COLS[d]] = "#4CAF50"

FEAT_LABELS = {
    "z_score":       "z-score",
    "log_z":         "log(1+z)",
    "harmonics":     "harmonics",
    "freq_deviation":"freq deviation",
    "sat_elevation": "elevation (°)",
    "hour_sin":      "hour sin",
    "hour_cos":      "hour cos",
    "campaign_day":  "campaign day",
    "elev_rate":     "elev. rate (°/s)",
}

names_rev  = list(feature_ls.keys())[::-1]
vals_rev   = [feature_ls[n] for n in names_rev]
colors_rev = [FEAT_COLORS[n] for n in names_rev]
labels_rev = [FEAT_LABELS.get(n, n) for n in names_rev]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(labels_rev, vals_rev, color=colors_rev, edgecolor="white", height=0.6)
ax.set_xlabel("Learned RBF lengthscale  (lower → more informative)")
ax.set_title("Per-Feature ARD Lengthscales from Additive GP Kernel")
ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.4, lw=0.9)

legend_els = [
    mpatches.Patch(facecolor="#FF5722", label="Spectral block"),
    mpatches.Patch(facecolor="#2196F3", label="Geometric block"),
    mpatches.Patch(facecolor="#4CAF50", label="Temporal block"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(FIGS / "lengthscales.png", dpi=150)
plt.close()
print("  Saved results/figures/lengthscales.png")

# ECE
p_va, _   = predict(model, lik, scaler, X_va)
p_te, s_te = predict(model, lik, scaler, X_te)
p_soft, s_soft = predict(model, lik, scaler, X_soft)
m_te_new  = compute_all(y_te, p_te)
m_va_new  = compute_all(y_va, p_va)
print(f"  val_ECE={m_va_new['ECE']:.5f}  te_ECE={m_te_new['ECE']:.5f}  AUROC={m_te_new['AUROC']:.4f}")

# ── 2. ECE improvement ─────────────────────────────────────────────────────────
print("\n[2] ECE comparison...")
baseline_ece = json.load(open(RESULTS / "ablation.json"))["full"]["te_ece"]
delta = m_te_new["ECE"] - baseline_ece
print(f"  Baseline (no elev_rate, no ARD): te_ECE = {baseline_ece:.5f}")
print(f"  Enhanced  (+ elev_rate + ARD):   te_ECE = {m_te_new['ECE']:.5f}")
print(f"  Delta: {delta:+.5f}  ({'improved' if delta < 0 else 'no improvement'})")

# ── 3. SOFT probability histogram ─────────────────────────────────────────────
print("\n[3] SOFT probability histogram...")

# Get predictions on HARD + BG samples from test set (labelled)
hard_mask = y_te == 1
bg_mask   = y_te == 0
p_hard    = p_te[hard_mask]
p_bg      = p_te[bg_mask]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
bins = np.linspace(0, 1, 26)

for ax, probs, label, color, subtitle in zip(
    axes,
    [p_bg, p_soft, p_hard],
    ["BACKGROUND\n(label=0)", "SOFT\n(held-out)", "HARD\n(label=1)"],
    ["#78909C", "#FF9800", "#E53935"],
    [f"n={len(p_bg):,}", f"n={len(p_soft):,}", f"n={len(p_hard):,}"],
):
    ax.hist(probs, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"{label}\n{subtitle}", fontsize=10)
    ax.set_xlabel("GP detection probability")
    ax.set_ylabel("Window count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0.5, color="black", lw=0.9, ls="--", alpha=0.6)

fig.suptitle("GP Probability Distribution by Window Class", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(FIGS / "soft_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved results/figures/soft_histogram.png")
print(f"  SOFT: mean_prob={p_soft.mean():.3f}  frac>0.5={( p_soft>0.5).mean():.3f}")

# ── 4. Reliability diagram ─────────────────────────────────────────────────────
print("\n[4] Reliability diagram...")
baseline_metrics = json.load(open(RESULTS / "baseline_metrics.json"))

# Threshold "soft probability": sigmoid of (z-4) * 5
z_te = te["z_score"].values
p_thr = 1 / (1 + np.exp(-5 * (z_te - 4.0)))

def reliability_points(y_true, y_prob, n_bins=12):
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() < 5:
            continue
        accs.append(float(y_true[m].mean()))
        confs.append(float(y_prob[m].mean()))
        ns.append(int(m.sum()))
    return np.array(confs), np.array(accs), np.array(ns)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration", alpha=0.5)

for label, probs, color in [
    ("Threshold (sigmoid)", p_thr,  "#e74c3c"),
    ("GP (this work)",      p_te,   "#2196F3"),
]:
    cx, cy, cw = reliability_points(y_te, probs)
    sz = cw / cw.max() * 120
    ax.scatter(cx, cy, s=sz, color=color, zorder=3, label=label)
    ax.plot(cx, cy, color=color, alpha=0.5, lw=1.5)

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability Diagram (Calibration)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(FIGS / "reliability.png", dpi=150)
plt.close()
print("  Saved results/figures/reliability.png")

# ── 5. Ablation ECE bar chart ──────────────────────────────────────────────────
print("\n[5] Ablation ECE bar chart...")
abl = json.load(open(RESULTS / "ablation.json"))
abl_names = {
    "full":       "Full\n(geom+spec+time)",
    "no_elev":    "No elevation\n(spec+time)",
    "spec_only":  "Spec only\n(spec+time)",
    "single_rbf": "Single RBF\n(no additive struct.)",
}
abl_colors = {
    "full":       "#2196F3",
    "no_elev":    "#FF9800",
    "spec_only":  "#4CAF50",
    "single_rbf": "#E53935",
}
variants  = list(abl.keys())
ece_vals  = [abl[v]["te_ece"] * 1e4 for v in variants]
col_list  = [abl_colors[v] for v in variants]
lbl_list  = [abl_names[v] for v in variants]

fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(lbl_list, ece_vals, color=col_list, edgecolor="white", width=0.55)
for bar, val in zip(bars, ece_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.2f}×10⁻⁴", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Test ECE (×10⁻⁴)")
ax.set_title("Ablation Study: Calibration Error by Kernel Variant")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(FIGS / "ablation_ece.png", dpi=150)
plt.close()
print("  Saved results/figures/ablation_ece.png")

# ── 6. System architecture figure ─────────────────────────────────────────────
print("\n[6] System architecture figure...")

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")

boxes = [
    (0.3, "SDR Signal\n(50 MS/s)", "#CFD8DC"),
    (1.9, "Welch PSD\n(Ku-band)", "#CFD8DC"),
    (3.5, "Feature\nEngineering\n(z, elev_rate…)", "#B3E5FC"),
    (5.2, "SVGP\nAdditive Kernel\nk_geom+k_spec+k_time", "#C8E6C9"),
    (7.0, "Bernoulli\nLikelihood\n+ELBO", "#C8E6C9"),
    (8.7, "P(detection)\n± uncertainty", "#FFE0B2"),
]

for x, label, color in boxes:
    rect = mpatches.FancyBboxPatch(
        (x, 0.7), 1.35, 1.6,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#455A64", linewidth=1.2
    )
    ax.add_patch(rect)
    ax.text(x + 0.675, 1.5, label, ha="center", va="center",
            fontsize=8.5, multialignment="center")

for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + 1.35
    x_end   = boxes[i+1][0]
    ax.annotate("", xy=(x_end, 1.5), xytext=(x_start, 1.5),
                arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.5))

ax.text(5.0, 0.15,
        "Inducing points (k-means, n=300) · CholeskyVariationalDistribution · StandardScaler",
        ha="center", va="bottom", fontsize=7.5, color="#546E7A", style="italic")

ax.set_title("StarLink GP Detector — System Overview", fontsize=11, pad=4)
plt.tight_layout()
fig.savefig(FIGS / "system_arch.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved results/figures/system_arch.png")

# ── 7. ECE comparison bar chart (all models) ───────────────────────────────────
print("\n[7] Full model ECE comparison...")
model_ece = {
    "Threshold":  0.0654,
    "LogReg":     0.000305,
    "MLP":        0.00606,
    "GP (Phase2)": 6.98e-5,
    "GP+elev_rate\n(Phase3+)": m_te_new["ECE"],
}
colors_model = ["#e74c3c", "#f39c12", "#27ae60", "#2196F3", "#0D47A1"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(list(model_ece.keys()), list(model_ece.values()),
              color=colors_model, edgecolor="white", width=0.6)
ax.set_yscale("log")
ax.set_ylabel("ECE (log scale, lower is better)")
ax.set_title("Expected Calibration Error — All Models")
for bar, val in zip(bars, model_ece.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.4,
            f"{val:.1e}", ha="center", va="bottom", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(FIGS / "ece_all_models.png", dpi=150)
plt.close()
print("  Saved results/figures/ece_all_models.png")

# ── 8. Pass-level aggregation ──────────────────────────────────────────────────
print("\n[8] Pass-level aggregation...")

te_full = te.reset_index(drop=True)
te_full["gp_prob"]       = p_te
te_full["threshold_det"] = (te_full["z_score"] >= 4.0).astype(int)
te_full["label"]         = y_te

time_gap           = te_full["unix_time"].diff()
new_pass           = (te_full["sat_name"] != te_full["sat_name"].shift()) | \
                     (time_gap > 30) | time_gap.isna()
te_full["pass_id"] = new_pass.cumsum()

pass_gp = te_full.groupby("pass_id").agg(
    gp_median=("gp_prob",       "median"),
    gp_max   =("gp_prob",       "max"),
    thr_any  =("threshold_det", "max"),
    true_pos =("label",         "max"),
    n_windows=("label",         "count"),
).reset_index()

pass_gp["gp_det_median"] = (pass_gp["gp_median"] > 0.5).astype(int)
pass_gp["gp_det_max"]    = (pass_gp["gp_max"]    > 0.5).astype(int)

def prf(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4),
            "tp": tp, "fp": fp, "fn": fn}

r_thr    = prf(pass_gp["true_pos"], pass_gp["thr_any"])
r_gp_med = prf(pass_gp["true_pos"], pass_gp["gp_det_median"])
r_gp_max = prf(pass_gp["true_pos"], pass_gp["gp_det_max"])

print(f"  Passes: {len(pass_gp):,}  Positive: {int(pass_gp['true_pos'].sum()):,}")
print(f"  Threshold: P={r_thr['precision']:.3f} R={r_thr['recall']:.3f} F1={r_thr['f1']:.3f}")
print(f"  GP median: P={r_gp_med['precision']:.3f} R={r_gp_med['recall']:.3f} F1={r_gp_med['f1']:.3f}")
print(f"  GP max:    P={r_gp_max['precision']:.3f} R={r_gp_max['recall']:.3f} F1={r_gp_max['f1']:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
methods = ["Threshold\n(any≥4.0)", "GP median\n(>0.5)", "GP max\n(>0.5)"]
prec    = [r_thr["precision"], r_gp_med["precision"], r_gp_max["precision"]]
rec     = [r_thr["recall"],    r_gp_med["recall"],    r_gp_max["recall"]]
f1s     = [r_thr["f1"],        r_gp_med["f1"],        r_gp_max["f1"]]
x       = np.arange(3); w = 0.22
ax.bar(x - w, prec, w, label="Precision", color="#2196F3", edgecolor="white")
ax.bar(x,     rec,  w, label="Recall",    color="#FF5722", edgecolor="white")
ax.bar(x + w, f1s,  w, label="F1",        color="#4CAF50", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.set_title("Pass-Level Detection Performance")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(FIGS / "pass_level.png", dpi=150)
plt.close()
print("  Saved results/figures/pass_level.png")

# ── Save full results ──────────────────────────────────────────────────────────
results = {
    "feature_lengthscales": feature_ls,
    "ece_comparison": {
        "baseline_te_ece":  baseline_ece,
        "enhanced_te_ece":  m_te_new["ECE"],
        "enhanced_val_ece": m_va_new["ECE"],
        "enhanced_auroc":   m_te_new["AUROC"],
        "delta":            delta,
    },
    "soft_analysis": {
        "mean_prob":       float(p_soft.mean()),
        "std_prob":        float(p_soft.std()),
        "mean_uncertainty":float(s_soft.mean()),
        "frac_above_half": float((p_soft > 0.5).mean()),
        "n_soft":          len(p_soft),
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
print("\nAll done. Figures in results/figures/")
