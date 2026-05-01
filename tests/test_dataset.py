"""Basic sanity checks for the data pipeline."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from gp_detector.dataset import engineer_features, make_labeled_split, \
                                chronological_split, get_arrays, FEATURE_COLS


def _make_dummy_df(n=200):
    rng = np.random.default_rng(0)
    conf = np.array(["BACKGROUND"] * 140 + ["SOFT"] * 40 + ["HARD"] * 20)
    rng.shuffle(conf)
    return pd.DataFrame({
        "id": np.arange(n),
        "timestamp_utc": pd.date_range("2026-02-21", periods=n, freq="30s").astype(str),
        "unix_time": np.linspace(1e9, 1e9 + n * 30, n),
        "z_score": rng.normal(2.5, 1.0, n),
        "peak_freq_hz": rng.normal(750, 30, n),
        "harmonics": rng.integers(0, 4, n),
        "sat_name": ["STARLINK-1"] * n,
        "sat_elevation": rng.uniform(50, 90, n),
        "confidence": conf,
    })


def test_feature_engineering():
    df = engineer_features(_make_dummy_df())
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"
    assert not df["log_z"].isna().any()
    assert not df["hour_sin"].isna().any()


def test_labeled_split_shapes():
    df = engineer_features(_make_dummy_df())
    train_df, soft_df = make_labeled_split(df, bg_ratio=2)
    assert set(train_df["label"].unique()).issubset({0, 1})
    assert len(soft_df) == (df["confidence"] == "SOFT").sum()


def test_chronological_split_no_leak():
    df = engineer_features(_make_dummy_df())
    train_df, _ = make_labeled_split(df)
    tr, va, te = chronological_split(train_df)
    assert tr["unix_time"].max() <= va["unix_time"].min()
    assert va["unix_time"].max() <= te["unix_time"].min()


def test_get_arrays_dtype():
    df = engineer_features(_make_dummy_df())
    train_df, _ = make_labeled_split(df)
    X, y = get_arrays(train_df)
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert X.shape[1] == len(FEATURE_COLS)
