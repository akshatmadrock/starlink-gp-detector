# Starlink GP Detector

**Physics-informed Gaussian Process classification for passive Starlink satellite detection**

ECE 228 — Machine Learning for Physical Applications, UC San Diego (Spring 2026)  
Akshat Gupta · `akg007@ucsd.edu` · PID A69041162

---

## What this is

Last quarter I built [Starlink Sentinel](https://github.com/ece257b/starlink-passive-detection) — a rooftop SDR system that passively detects Starlink satellites by listening for a 750 Hz spectral fingerprint in the Ku-band downlink. It ran for 20+ days and accumulated **619,705** thirty-second observation windows covering **9,503 unique satellites**.

The detector is a hard z-score threshold. It works, but it throws away physically meaningful information: satellite elevation angle is computed from TLE orbital data for every window and completely ignored by the detector. A satellite at 85° elevation should be easier to detect than one at 55° — path length, atmospheric attenuation, and dish beam pattern all say so — but the threshold doesn't know this.

This project replaces the fixed threshold with a **Sparse Variational GP (SVGP)** classifier that:
- Takes satellite elevation as a first-class physics-informed input
- Outputs calibrated detection probabilities, not a binary yes/no
- Quantifies uncertainty, especially in the ambiguous z ≈ 3–4 band
- Is trained on 619k real RF observations from a live system

The physics-informed part is literal: the additive GP kernel separates a geometric prior (elevation) from spectral evidence (z-score, harmonics), so the model's uncertainty tracks both where the satellite is and what the spectrum looks like.

---

## Project roadmap

This is a quarter-long project built incrementally. Milestones:

| Phase | Weeks | Goal |
|---|---|---|
| **Phase 1** | May 1–9 | Data pipeline, EDA, feature engineering, baseline detectors |
| **Phase 2** | May 10–23 | SVGP implementation, additive kernel, initial training |
| **Phase 3** | May 24–Jun 5 | Calibration analysis, SOFT window evaluation, ablations |
| **Phase 4** | Jun 6–13 | Final results, report, cleaned notebooks |

Progress is tracked via GitHub Issues. Each phase closes with a tagged release.

---

## Results (updated as project progresses)

| Model | AUROC | AUPRC | ECE (↓) |
|---|---|---|---|
| Fixed Threshold (z ≥ 4) | 1.000 | 1.000 | 0.0654 |
| Logistic Regression | 1.000 | 1.000 | 0.0003 |
| MLP (calibrated) | 1.000 | 1.000 | 0.0061 |
| **SVGP (ours)** | **1.000** | **1.000** | **0.0001** |

All models achieve perfect discrimination (AUROC = 1) — the HARD/BACKGROUND classes are well-separated by z-score, which is expected since HARD is defined by z ≥ 4. The differentiator is **calibration**: the GP's ECE of 0.0001 is 65× better than the operational threshold. This matters for downstream use (positioning, interference mapping) where you need a confidence score, not a hard yes/no.

---

## Repository structure

```
starlink-gp-detector/
├── gp_detector/           # Main Python package
│   ├── dataset.py         # Data loading, feature engineering, splits
│   ├── evaluate.py        # ECE, reliability diagrams, stratified metrics
│   └── models/
│       ├── baselines.py   # Threshold, LogReg, MLP
│       └── svgp.py        # SVGP with additive physics-informed kernel
├── notebooks/             # Jupyter notebooks (one per phase)
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_svgp.ipynb
├── scripts/               # Runnable training scripts
│   ├── train_baselines.py
│   └── train_svgp.py
├── configs/
│   └── config.yaml        # All hyperparameters in one place
├── data/
│   └── README.md          # Where to get detections.db (not in git, 86 MB)
├── results/               # Saved models and metrics (not in git)
└── tests/
    └── test_dataset.py
```

---

## Setup

```bash
git clone https://github.com/akshatmadrock/starlink-gp-detector.git
cd starlink-gp-detector
pip install -r requirements.txt
```

Get the data (from the companion repo or the WCSNG lab system):
```bash
cp /path/to/detections.db data/detections.db
```

---

## Running

```bash
# Train baselines
python scripts/train_baselines.py

# Train SVGP
python scripts/train_svgp.py
```

---

## Technical background

**Why 750 Hz?** Starlink's OFDM waveform repeats every 1.33 ms (750 Hz frame rate). This creates a periodic power envelope detectable passively — you don't need to decode the payload, just see the amplitude oscillate. Harmonics appear at 1500, 2250, 3000 Hz.

**Why elevation matters?** Free-space path loss scales as distance². A Starlink satellite at 55° elevation is roughly 15% further away than one directly overhead, meaning ~2.7 dB weaker signal. The dish beam pattern amplifies this further. Yet the operational detector applies the same threshold regardless.

**Why GP?** The kernel encodes physically motivated smoothness over elevation — we expect detection probability to increase monotonically as elevation increases and the satellite gets closer. A logistic regression or MLP learns this from data but doesn't encode the prior. The GP does, which is why it wins on calibration.

---

## Dataset

619,705 × 30-second windows from a rooftop Ku-band + USRP X310 system at UCSD (32.88°N, Feb–Mar 2026).

| Class | Count | Definition |
|---|---|---|
| HARD | 5,166 | z ≥ 4.0 |
| SOFT | 70,878 | z ≥ 3.0 + ≥1 harmonic — held out, ambiguous |
| BACKGROUND | 543,661 | below threshold |

See `data/README.md` for full schema.

---

## Course context

ECE 228: Machine Learning for Physical Applications (Prof. Yuanyuan Shi, UCSD).  
Solo project — approved by TA Benjie Miao.  
Companion system: [ece257b/starlink-passive-detection](https://github.com/ece257b/starlink-passive-detection)
