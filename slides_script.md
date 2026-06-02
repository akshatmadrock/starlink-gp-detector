# Presentation Script (2 minutes total)

---

## Slide 1 -- Title (~15 seconds)

"Hi, I'm Akshat, this is Team 50.
So this project started in ECE 257B last quarter, where I built a passive Starlink
detector using an SDR and a z-score threshold.
It worked, but I had two problems with it.
This project, for ECE 228, is me fixing those problems with a Gaussian Process."

---

## Slide 2 -- Prior Work + Problem (~25 seconds)

"The 257B setup was: record Ku-band at 50 megasamples per second, compute a Welch
periodogram, and fire when the 750 Hz bin is unusually elevated compared to the
local noise floor.

The ranking was essentially perfect. But the output is binary, there's no confidence
score, and about 70,000 windows that fall in the ambiguous z-score range just get
thrown out completely. For any kind of track-before-detect application, those
borderline windows are actually the most useful ones, and the threshold just
discards them."

---

## Slide 3 -- The Approach (~30 seconds)

"So I replaced the threshold with a sparse variational GP with a Bernoulli likelihood.

The key design choice is the additive kernel. Instead of one RBF over all features,
I split the covariance into three physically-motivated blocks: a geometric block
that gets elevation and angular velocity, a spectral block for the z-score and
harmonic features, and a temporal block for time-of-day and campaign drift.

Angular velocity, d-theta-d-t, is a feature I added for ECE 228: a fast-moving
satellite causes more Doppler smearing of the 750 Hz peak, so z-score becomes
less reliable even at the same elevation. The geometric block needed that kinematic
signal to actually do anything useful.

Each block has ARD, so the learned lengthscales tell you which features the model
is using."

---

## Slide 4 -- Results (~25 seconds)

"The main result is calibration. Expected Calibration Error is 230 times lower
than the threshold detector, and that number is stable across 5 random seeds.

The ablation says the additive structure is what matters: a single RBF over all
features is 5.7 times worse in ECE despite identical ranking. The structure of
the kernel is load-bearing, not just decorative.

For the 70,000 soft windows the threshold discards: GP probabilities in satellite
passes that are confirmed positive are at 52.9% above 0.5, versus 18% in negative
passes. The GP is doing real work on data it never trained on."

---

## Slide 5 -- Takeaway (~25 seconds)

"Three quick points to close.

First, kernel design is the model. You can see this in the ablation: structuring
the covariance to match the physical evidence sources matters more than the
choice of model class.

Second, calibration and discrimination are not the same thing. AUROC is 1.0 for
every model here because the labels are defined by z-score. ECE is the only number
that tells you anything interesting.

Third, the 70,000 windows the threshold throws away are now useful. That is the
whole point of replacing it.

Thanks."

---

## Timing check

- Slide 1: 15s
- Slide 2: 25s
- Slide 3: 30s
- Slide 4: 25s
- Slide 5: 25s
- Total: ~2 min flat

## Tips

- Move through slides at a steady pace, don't wait for questions mid-slide
- Slide 3 is the longest, you can cut the angular velocity detail if running long
- The ablation number (5.7x) is the sharpest line, say it clearly
- Don't say "um" before "230 times" -- just say the number
