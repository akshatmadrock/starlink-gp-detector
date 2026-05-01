"""
Three baselines for comparison against the GP detector.

  1. ThresholdDetector    — the current operational system (z >= 4.0)
  2. LogisticRegression   — linear probabilistic classifier
  3. MLP                  — nonlinear classifier, isotonic-calibrated
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


class ThresholdDetector:
    """Soft-probability wrapper around the existing hard z-score threshold."""

    def __init__(self, threshold: float = 4.0, z_col_idx: int = 0):
        self.threshold = threshold
        self.z_col_idx = z_col_idx

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z    = X[:, self.z_col_idx]
        prob = 1.0 / (1.0 + np.exp(-2.0 * (z - self.threshold)))
        return np.stack([1 - prob, prob], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.z_col_idx] >= self.threshold).astype(int)


def build_logreg(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=C, max_iter=1000, class_weight="balanced")),
    ])


def build_mlp(hidden: tuple = (64, 32), seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
            method="isotonic",
            cv=3,
        )),
    ])
