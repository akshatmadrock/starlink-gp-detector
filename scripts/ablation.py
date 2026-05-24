"""
Ablation study: quantifies the contribution of each component.

Variants trained:
  full       — additive kernel (geom + spec + time), all features [baseline]
  no_elev    — additive (spec + time), elevation zeroed out
  single_rbf — single RBF on all features, no additive structure
  spec_only  — spectral + time features only (identical to no_elev structurally)

For each variant we report val/test ECE and AUROC, plus GP probs on SOFT windows.
Results saved to results/ablation.json

Run from repo root:  python scripts/ablation.py
"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from gp_detector.dataset     import load_raw, engineer_features, make_labeled_split, \
                                    chronological_split, get_arrays, FEATURE_COLS
from gp_detector.models.svgp import train, predict
from gp_detector.evaluate    import compute_all, stratified, ELEV_BINS, ELEV_LABELS

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

print("Loading data...")
df, soft_df = make_labeled_split(engineer_features(load_raw()))
tr, va, te  = chronological_split(df)
X_tr, y_tr  = get_arrays(tr)
X_va, y_va  = get_arrays(va)
X_te, y_te  = get_arrays(te)
elev_te     = te["sat_elevation"].values
X_soft      = soft_df[FEATURE_COLS].values.astype("float32")
elev_soft   = soft_df["sat_elevation"].values
print(f"  Train {len(X_tr):,} | Val {len(X_va):,} | Test {len(X_te):,} | Soft {len(X_soft):,}")

VARIANTS = ["full", "no_elev", "single_rbf", "spec_only"]
ablation_results = {}

for variant in VARIANTS:
    print(f"\n[{variant}] training...")
    t0 = time.time()
    model, lik, scaler, losses = train(
        X_tr, y_tr,
        n_inducing=200,
        n_epochs=60,
        lr=0.05,
        variant=variant,
        verbose=False,
    )
    elapsed = time.time() - t0

    p_va,  _    = predict(model, lik, scaler, X_va)
    p_te,  s_te = predict(model, lik, scaler, X_te)
    p_soft, s_soft = predict(model, lik, scaler, X_soft)

    m_va  = compute_all(y_va, p_va)
    m_te  = compute_all(y_te, p_te)
    strat = stratified(y_te, p_te, elev_te)

    # SOFT window stats
    soft_stats = {
        "mean_prob": float(p_soft.mean()),
        "std_prob":  float(p_soft.std()),
        "mean_uncertainty": float(s_soft.mean()),
        "frac_above_half": float((p_soft > 0.5).mean()),
        # Mean GP prob per elevation bin on SOFT
        "by_elevation": {}
    }
    for (lo, hi), lbl in zip(ELEV_BINS, ELEV_LABELS):
        m = (elev_soft >= lo) & (elev_soft < hi)
        if m.sum() < 10:
            continue
        soft_stats["by_elevation"][lbl] = {
            "mean_prob": float(p_soft[m].mean()),
            "mean_uncertainty": float(s_soft[m].mean()),
            "n": int(m.sum()),
        }

    ablation_results[variant] = {
        "val_ece":   m_va["ECE"],
        "val_auroc": m_va["AUROC"],
        "te_ece":    m_te["ECE"],
        "te_auroc":  m_te["AUROC"],
        "final_loss": losses[-1],
        "time_s":    round(elapsed, 1),
        "stratified": strat,
        "soft":      soft_stats,
    }

    print(f"  val_ECE={m_va['ECE']:.5f}  te_ECE={m_te['ECE']:.5f}  "
          f"AUROC={m_te['AUROC']:.4f}  ({elapsed:.0f}s)")
    print(f"  SOFT: mean_prob={p_soft.mean():.3f}  mean_uncertainty={s_soft.mean():.3f}")

json.dump(ablation_results, open(RESULTS / "ablation.json", "w"), indent=2)

print("\n=== Ablation Summary ===")
print(f"{'Variant':<14} {'val_ECE':>9} {'te_ECE':>9} {'AUROC':>7} {'SOFT_mean_p':>12} {'SOFT_unc':>10}")
print("-" * 65)
for v, r in ablation_results.items():
    print(f"{v:<14} {r['val_ece']:>9.5f} {r['te_ece']:>9.5f} "
          f"{r['te_auroc']:>7.4f} {r['soft']['mean_prob']:>12.3f} "
          f"{r['soft']['mean_uncertainty']:>10.3f}")

print("\nSaved to results/ablation.json")
