# Methodology

## The question

The core of the mammalian circadian clock is a negative-feedback loop: CLOCK and BMAL1 switch on the *Per*
and *Cry* genes, and the PER and CRY proteins, after a delay, switch them off again. A loop like this does not
oscillate automatically. Left to itself it settles to a steady level unless the repression is both strong and
late enough. There are three textbook explanations of what supplies that:

1. **Steep repression.** In Goodwin's model (1965) each stage is removed in proportion to its amount, and the
   loop only oscillates if repression is extremely cooperative: a Hill coefficient above 8 with three stages
   (Griffith, 1968), which is far steeper than any known transcriptional repressor.
2. **Saturating removal.** If the enzymes that degrade mRNA and protein saturate, removal is no longer
   proportional to amount, and the loop can oscillate with gentle repression (Gonze et al., 2005).
3. **A longer delay.** With more intermediate steps between transcription and repression (phosphorylation,
   complex formation, nuclear entry), gentle repression is enough: with seven stages, a Hill coefficient above
   about 2.1 (the secant condition; Tyson and Othmer, 1978).

These are different claims about how the clock works, and they make different predictions about drugs and
mutations. But each can be tuned to produce the same daily rhythm. This project asks: **which experiment would
tell them apart?** And, within whichever model is right, **how much can the usual data say about its
parameters?**

## The models

For stages *i* = 1..*N*, with *x*<sub>1</sub> the mRNA and *x*<sub>N</sub> the repressor:

```
dx_1/dt = v / (1 + x_N^n) - removal_1(x_1)
dx_i/dt = k_i x_(i-1) - removal_i(x_i)          i = 2..N
```

| Model | Stages | Removal | Free parameters |
|---|---|---|---|
| Steep switch | 3 | r<sub>i</sub> x | three removal rates, Hill coefficient *n* (4 to 60), gain |
| Saturating removal | 3 | V<sub>i</sub> x / (K<sub>m</sub> + x) | transcription *v*, *n* (1 to 4), conversion rate *k*, three V<sub>i</sub>, K<sub>m</sub> |
| Long delay chain | 7 | r<sub>i</sub> x | mRNA removal rate, one rate for all six protein steps, *n* (1 to 4), gain |

Concentrations are in units where repression is half-maximal at *x*<sub>N</sub> = 1, and time is in hours. In
the two linear models each protein stage is made at the rate it is removed (*k*<sub>i</sub> = *r*<sub>i</sub>),
which only fixes the units of the intermediate stages, and *v* = gain × *r*<sub>1</sub>, so the gain is the level
the loop would reach without repression. Multiplying every rate by the same factor then speeds the clock up by
that factor and leaves the waveform unchanged.

The bounds on the Hill coefficient keep the hypotheses distinct. If the saturation model were allowed steep
repression, it would contain the switch as a special case (removal far below saturation is linear), and the
delay model would come close to it too. The question is what supplies the nonlinearity *when repression is not
steep*, so their Hill coefficients are limited to 4.

**Base parameters.** Each model has a base parameter set (`scripts/tune_models.py`) with a 24-hour period
and the same 46-fold peak-to-trough swing in mRNA (3.84 on the natural log scale). The saturation model is the
least flexible: it keeps oscillating through every perturbation in the experiments only in a band of parameters
where its amplitude hardly changes, so it was set by hand and the other two were tuned to match it, each by
bisection on its gain, then rescaled in time to exactly 24 hours. Matching the swing took a Hill coefficient of
20 in the switch, 2 with saturating removal and 3 with the delay chain. These base sets are the "truths" in the
design analysis and the simulation study.

**Perturbations.** Two stand for real experiments:

* an **inhibitor** multiplies the removal rate of every protein stage by *f*, as a CK1δ/ε inhibitor slows the
  phosphorylation-dependent degradation of PER (*f* = 0.9, 0.8 and 0.7 below);
* **half dosage** multiplies the transcription rate by 0.5, as in a heterozygous knockout.

**Computing the rhythm.** The equations are integrated by fourth-order Runge–Kutta with a step of 0.05 hours
(compiled with numba), until two successive cycles agree in period to 0.1% and the amplitude has stopped
shrinking. The period comes from the times log mRNA rises through its midpoint, interpolated between steps. If
the oscillation decays, or no cycle appears, the model is recorded as not oscillating. Starting each
integration from the cycle of a nearby parameter set makes repeated evaluations during fitting fast (about 4 ms
each).

## What is measured

Every experiment includes the **standard time course**: mRNA every 4 hours for two days (13 samples), each
with a measurement error of 0.15 on the log scale (about 15%). Levels are only known relative to an unknown
scale, as from a microarray or qPCR, so each time course has a free offset. The other experiments add to it:

| Experiment | Adds | Error |
|---|---|---|
| Hourly time course | mRNA every hour instead (49 samples) | 0.15 |
| Add protein | total protein (all stages after mRNA) at the same 13 times | 0.20 |
| Add an inhibitor series | period and amplitude of a reporter in control and at *f* = 0.9, 0.8, 0.7 | 0.25 h; 0.10 |
| Add half dosage | period and amplitude of a reporter in control and with transcription halved | 0.25 h; 0.10 |
| Everything | hourly mRNA, protein, the inhibitor series and half dosage | |

Period and amplitude measurements stand for luciferase recordings of cells or tissue slices, which give the
free-running period to within a fraction of an hour. Amplitude is recorded as the log of the perturbed
amplitude relative to the control.

## Fitting

With Gaussian measurement errors of known size, minus twice the log likelihood is the sum of squared
standardised residuals (SSE) plus a constant, so the best fit is a weighted least-squares fit. Parameters are
fitted on the log scale within each model's bounds, together with the phase of the clock at the first sample.
Each time course's offset is profiled out: for any parameters, the best offset is the mean difference between
data and prediction. Fits use scipy's trust-region reflective least squares with forward-difference
derivatives.

A model that does not oscillate where the data show a rhythm predicts flat time courses, and each rhythm
measurement it cannot predict adds a penalty of 20 standard deviations. This keeps the search away from
parameters that stop the clock without leaving the objective undefined there.

The objective has many local minima, so each fit runs from several starting points: the model's base
parameters, random draws around them (log-normal, sd 0.5) and, where available, the model's fits to related
data. Every start is first rescaled in time to the period seen in the data and given the best phase on a grid.

Models are compared by BIC = SSE + *k* log *n*, where *k* counts the model's parameters, the phase and the
offsets, and *n* is the number of observations.
