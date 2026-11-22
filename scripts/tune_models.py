"""Find each model's base parameter set: a 24 h period and the same log mRNA amplitude.

The saturation model is the least flexible: it oscillates robustly (through every perturbation in the
experiments) only in a band where its amplitude barely changes with its parameters. So its parameters are
fixed by hand, and the other two models are matched to its amplitude: each has one knob (gain) that mostly
sets the amplitude, found by bisection on the log scale. Finally every rate is rescaled so the period is
exactly 24 h, which leaves the amplitude unchanged. Prints the parameter sets to paste into models.py, and
checks that each still oscillates under every perturbation used in the experiments.

Usage:  python scripts/tune_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clocks.experiments import PERTURBATIONS  # noqa: E402
from clocks.models import MODELS, limit_cycle, rescale_time  # noqa: E402

# starting parameters, and the knob that is tuned (with the range searched) or None
START = {
    "saturation": ((0.7, 2.0, 0.6, 0.3, 0.3, 0.3, 0.2), None),
    "switch": ((0.24, 0.2, 0.16, 20.0, 30.0), ("gain", 5.0, 5000.0)),
    "delay": ((0.55, 0.55, 3.0, 10.0), ("gain", 5.0, 200.0)),
}


def amplitude(model, theta):
    lc = limit_cycle(model, theta)
    return -1.0 if lc is None else lc.amplitude


def tune(key, target=None):
    model = MODELS[key]
    start, knob = START[key]
    theta = np.array(start, dtype=float)
    if knob is not None:
        name, a, b = knob
        j = model.param_names.index(name)
        lo, hi = np.log(a), np.log(b)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            theta[j] = np.exp(mid)
            if amplitude(model, theta) < target:
                lo = mid
            else:
                hi = mid
        theta[j] = np.exp(0.5 * (lo + hi))
    lc = limit_cycle(model, theta)
    theta = rescale_time(model, theta, lc.period / 24.0)
    return theta, limit_cycle(model, theta)


def main():
    theta, lc = tune("saturation")
    target = lc.amplitude
    results = {"saturation": (theta, lc)}
    for key in ("switch", "delay"):
        results[key] = tune(key, target)
    for key in MODELS:
        theta, lc = results[key]
        model = MODELS[key]
        print(f"{key:11s} period {lc.period:.3f} h, log mRNA amplitude {lc.amplitude:.3f}, "
              f"log protein amplitude {np.ptp(lc.log_protein):.3f}")
        print("   base=(" + ", ".join(f"{x:.4g}" for x in theta) + "),")
        for pert in PERTURBATIONS:
            p = limit_cycle(model, theta, pert, warm=lc.state)
            print(f"   {pert.label:28s} " + ("no oscillation" if p is None else
                  f"period {p.period:5.1f} h, amplitude x{p.amplitude / lc.amplitude:.2f}"))


if __name__ == "__main__":
    main()
