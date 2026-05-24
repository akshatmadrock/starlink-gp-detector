"""
Data loading, feature engineering, and splitting for the GP Starlink detector.

Label strategy
--------------
  HARD       -> 1   confident positive  (z >= 4.0, defined by the classical detector)
  BACKGROUND -> 0   confident negative  (below-threshold windows)
  SOFT       -> held out               ambiguous zone used only for calibration analysis

Feature engineering
-------------------
  z_score          raw Welch PSD z-score at 750 Hz
  log_z            log1p(z_score) — better numerical range for GP
  harmonics        confirmed harmonic count (0–3), clipped
  freq_deviation   |peak_freq - 750| / 50, how far the peak drifted
  sat_elevation    elevation angle of overhead Starlink satellite (TLE-derived)
  hour_sin/cos     sine-cosine encoding of hour-of-day (diurnal pattern)
  campaign_day     days since campaign start (captures slow hardware drift)
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# Default path relative to repo root when running scripts from scripts/
_DEFAULT_DB = Path(__file__).parents[2] / "starlink-passive-detection" / "data" / "detections.db"

FEATURE_COLS = [
    "z_score",          # 0
    "log_z",            # 1
    "harmonics",        # 2
    "freq_deviation",   # 3
    "sat_elevation",    # 4
    "hour_sin",         # 5
    "hour_cos",         # 6
    "campaign_day",     # 7
    "elev_rate",        # 8  angular velocity (deg/s) — encodes Doppler broadening
]


def load_raw(db_path: Path = _DEFAULT_DB) -> pd.DataFrame:
    """Load every row from detections.db."""
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query("SELECT * FROM detections", conn)
    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp_utc"])

    hour           = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    t0                  = df["unix_time"].min()
    df["campaign_day"]  = (df["unix_time"] - t0) / 86400.0
    df["freq_deviation"] = np.abs(df["peak_freq_hz"] - 750.0) / 50.0
    df["harmonics"]     = df["harmonics"].clip(0, 3)
    df["log_z"]         = np.log1p(df["z_score"].clip(lower=0))

    # Angular velocity (deg/s): dθ/dt within each satellite pass.
    # Encodes Doppler broadening — fast-moving satellites cause PSD smearing.
    # Pass boundary: same sat_name + time gap < 30 s.
    df = df.sort_values("unix_time").reset_index(drop=True)
    time_gap    = df["unix_time"].diff()
    new_pass    = (df["sat_name"] != df["sat_name"].shift()) | (time_gap > 30) | time_gap.isna()
    df["_pass"] = new_pass.cumsum()
    g = df.groupby("_pass", sort=False)
    df["elev_rate"] = (
        g["sat_elevation"].diff() / g["unix_time"].diff()
    ).fillna(0.0)
    df.drop(columns="_pass", inplace=True)

    return df


def make_labeled_split(df: pd.DataFrame, bg_ratio: int = 10, seed: int = 42):
    """
    Returns (train_df, soft_df).

    train_df  — HARD (label=1) + downsampled BACKGROUND (label=0)
    soft_df   — all SOFT windows, held out for calibration analysis
    """
    hard = df[df["confidence"] == "HARD"].copy()
    soft = df[df["confidence"] == "SOFT"].copy()
    bg   = df[df["confidence"] == "BACKGROUND"].copy()

    hard["label"] = 1
    bg["label"]   = 0

    n_bg = min(len(bg), len(hard) * bg_ratio)
    bg   = bg.sample(n=n_bg, random_state=seed)

    train_df = pd.concat([hard, bg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, soft


def chronological_split(df: pd.DataFrame, val_frac: float = 0.15,
                        test_frac: float = 0.15) -> tuple:
    """
    Chronological train/val/test split.
    Uses time ordering to prevent future data leaking into training.
    """
    df  = df.sort_values("unix_time").reset_index(drop=True)
    n   = len(df)
    i_v = int(n * (1 - val_frac - test_frac))
    i_t = int(n * (1 - test_frac))
    return df.iloc[:i_v].copy(), df.iloc[i_v:i_t].copy(), df.iloc[i_t:].copy()


def get_arrays(df: pd.DataFrame) -> tuple:
    """Return (X, y) numpy arrays from a labeled dataframe."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y
