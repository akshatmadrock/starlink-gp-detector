"""
Hyperparameter sweep over n_inducing and learning rate.
Run from repo root:  python scripts/hparam_sweep.py

Sweeps n_inducing in [50, 100, 200, 300, 500] with fixed lr=0.05,
then sweeps lr in [0.01, 0.03, 0.05, 0.1] with best n_inducing.

Results saved to results/hparam_sweep.json
"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from gp_detector.dataset     import load_raw, engineer_features, make_labeled_split, \
                                    chronological_split, get_arrays
from gp_detector.models.svgp import train, predict
from gp_detector.evaluate    import compute_all

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

print("Loading data...")
df, _ = make_labeled_split(engineer_features(load_raw()))
tr, va, te = chronological_split(df)
X_tr, y_tr = get_arrays(tr)
X_va, y_va = get_arrays(va)
X_te, y_te = get_arrays(te)

sweep_results = []

# --- Sweep 1: n_inducing ------------------------------------------------
print("\n=== Sweep 1: n_inducing (lr=0.05, epochs=50) ===")
for n_ind in [50, 100, 200, 300, 500]:
    t0 = time.time()
    model, lik, scaler, losses = train(
        X_tr, y_tr, n_inducing=n_ind, n_epochs=50, lr=0.05, verbose=False
    )
    p_val, _ = predict(model, lik, scaler, X_va)
    p_te,  _ = predict(model, lik, scaler, X_te)
    m_val = compute_all(y_va, p_val)
    m_te  = compute_all(y_te,  p_te)
    elapsed = time.time() - t0
    entry = {
        "sweep": "n_inducing", "n_inducing": n_ind, "lr": 0.05,
        "val_ece": m_val["ECE"], "val_auroc": m_val["AUROC"],
        "te_ece":  m_te["ECE"],  "te_auroc":  m_te["AUROC"],
        "final_loss": losses[-1], "time_s": round(elapsed, 1),
    }
    sweep_results.append(entry)
    print(f"  n_inducing={n_ind:3d}  val_ECE={m_val['ECE']:.5f}  "
          f"te_ECE={m_te['ECE']:.5f}  {elapsed:.0f}s")

# Pick best n_inducing by val ECE
best_n = min([r for r in sweep_results if r["sweep"]=="n_inducing"],
             key=lambda r: r["val_ece"])["n_inducing"]
print(f"\nBest n_inducing: {best_n}")

# --- Sweep 2: learning rate ---------------------------------------------
print(f"\n=== Sweep 2: lr (n_inducing={best_n}, epochs=50) ===")
for lr in [0.01, 0.03, 0.05, 0.1]:
    t0 = time.time()
    model, lik, scaler, losses = train(
        X_tr, y_tr, n_inducing=best_n, n_epochs=50, lr=lr, verbose=False
    )
    p_val, _ = predict(model, lik, scaler, X_va)
    p_te,  _ = predict(model, lik, scaler, X_te)
    m_val = compute_all(y_va, p_val)
    m_te  = compute_all(y_te,  p_te)
    elapsed = time.time() - t0
    entry = {
        "sweep": "lr", "n_inducing": best_n, "lr": lr,
        "val_ece": m_val["ECE"], "val_auroc": m_val["AUROC"],
        "te_ece":  m_te["ECE"],  "te_auroc":  m_te["AUROC"],
        "final_loss": losses[-1], "time_s": round(elapsed, 1),
    }
    sweep_results.append(entry)
    print(f"  lr={lr:.2f}  val_ECE={m_val['ECE']:.5f}  "
          f"te_ECE={m_te['ECE']:.5f}  {elapsed:.0f}s")

json.dump(sweep_results, open(RESULTS / "hparam_sweep.json", "w"), indent=2)
print(f"\nSaved {len(sweep_results)} runs to results/hparam_sweep.json")
