"""Experiments: what is measured, how precisely, and what each model predicts for it.

An experiment (a Design) is made of two kinds of measurement:

  TimeCourse  log mRNA or log total protein sampled at fixed times in the unperturbed clock. Levels are only
              known relative to an unknown scale (a microarray or qPCR measures fold changes), so each time
              course has its own free offset, which is profiled out when fitting.
  Rhythm      the free-running period and amplitude of a reporter, in control conditions and under one or
              more perturbations, as from luciferase recordings of cells or tissue slices. Reported as the
              period in hours and the log of the amplitude relative to the control.

Every design includes the standard time course (mRNA every 4 h for two days), so each comparison asks what
an extra measurement adds to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import NONE, LimitCycle, Model, Perturbation, limit_cycle

SIGMA_LEVEL = 0.15     # sd of a log mRNA measurement (about 15%)
SIGMA_PROTEIN = 0.20   # sd of a log protein measurement (western blots are noisier)
SIGMA_PERIOD = 0.25    # sd of a measured period, hours
SIGMA_AMPLITUDE = 0.10  # sd of a measured log amplitude ratio

INHIBITOR_DOSES = (0.9, 0.8, 0.7)
PERTURBATIONS = tuple(Perturbation(inhibitor=f) for f in INHIBITOR_DOSES) + (Perturbation(dosage=0.5),)


@dataclass(frozen=True)
class TimeCourse:
    species: str               # "mrna" or "protein"
    times: tuple[float, ...]   # hours from the start of sampling
    sigma: float

    @property
    def n(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class Rhythm:
    perturbations: tuple[Perturbation, ...]
    sigma_period: float = SIGMA_PERIOD
    sigma_amplitude: float = SIGMA_AMPLITUDE

    @property
    def n(self) -> int:
        """Control period, then a period and an amplitude ratio for each perturbation."""
        return 1 + 2 * len(self.perturbations)


@dataclass(frozen=True)
class Design:
    key: str
    name: str
    description: str
    courses: tuple[TimeCourse, ...]
    rhythm: Rhythm | None = None

    @property
    def n_obs(self) -> int:
        return sum(c.n for c in self.courses) + (self.rhythm.n if self.rhythm else 0)

    @property
    def perturbations(self) -> tuple[Perturbation, ...]:
        return self.rhythm.perturbations if self.rhythm else ()

    @property
    def sigmas(self) -> np.ndarray:
        """Measurement sd of each observation, in the order of predict()."""
        parts = [np.full(c.n, c.sigma) for c in self.courses]
        if self.rhythm:
            r = self.rhythm
            parts.append([r.sigma_period])
            for _ in r.perturbations:
                parts.append([r.sigma_period, r.sigma_amplitude])
        return np.concatenate([np.asarray(p, dtype=float) for p in parts])

    def labels(self) -> list[str]:
        out = [f"{c.species} t={t:g}h" for c in self.courses for t in c.times]
        if self.rhythm:
            out.append("period, control")
            for p in self.rhythm.perturbations:
                out += [f"period, {p.label}", f"log amplitude ratio, {p.label}"]
        return out


def _hours(step: float, span: float = 48.0) -> tuple[float, ...]:
    return tuple(float(t) for t in np.arange(0.0, span + 1e-9, step))


STANDARD = TimeCourse("mrna", _hours(4.0), SIGMA_LEVEL)
HOURLY = TimeCourse("mrna", _hours(1.0), SIGMA_LEVEL)
PROTEIN = TimeCourse("protein", _hours(4.0), SIGMA_PROTEIN)

DESIGNS = {d.key: d for d in (
    Design("standard", "Standard time course",
           "mRNA every 4 hours for two days (13 samples).", (STANDARD,)),
    Design("hourly", "Hourly time course",
           "mRNA every hour for two days (49 samples), four times the sampling effort.", (HOURLY,)),
    Design("protein", "Add protein",
           "The standard time course plus total protein at the same 13 times.", (STANDARD, PROTEIN)),
    Design("inhibitor", "Add an inhibitor series",
           "The standard time course plus the period and amplitude of a reporter with protein removal slowed "
           "by 10%, 20% and 30%, as by a CK1 inhibitor.", (STANDARD,), Rhythm(PERTURBATIONS[:3])),
    Design("dosage", "Add half gene dosage",
           "The standard time course plus the period and amplitude of a reporter with transcription halved, "
           "as in a heterozygous knockout.", (STANDARD,), Rhythm(PERTURBATIONS[3:])),
    Design("both", "Add both perturbations",
           "The standard time course plus the inhibitor series and half dosage: three reporter experiments, "
           "no new time courses.", (STANDARD,), Rhythm(PERTURBATIONS)),
    Design("everything", "Everything",
           "Hourly mRNA, protein, the inhibitor series and half dosage together.",
           (HOURLY, PROTEIN), Rhythm(PERTURBATIONS)),
)}


@dataclass
class Prediction:
    """What a model predicts for a design. levels are the predicted log levels of each time course (before
    the free offset); rhythm is [control period, period_1, log amplitude ratio_1, ...], with NaN where
    the model does not oscillate."""
    levels: list[np.ndarray]
    rhythm: np.ndarray
    cycles: dict                 # perturbation -> LimitCycle or None (NONE is the unperturbed clock)

    @property
    def oscillates(self) -> bool:
        return self.cycles[NONE] is not None

    def vector(self) -> np.ndarray:
        return np.concatenate(self.levels + [self.rhythm])


def cycles_for(model: Model, theta, design: Design, warm: dict | None = None) -> dict:
    """The limit cycle under each condition the design needs; warm maps a condition to a starting state."""
    warm = warm or {}
    out = {NONE: limit_cycle(model, theta, NONE, warm=warm.get(NONE))}
    for p in design.perturbations:
        start = warm.get(p)
        if start is None and out[NONE] is not None:
            start = out[NONE].state
        out[p] = limit_cycle(model, theta, p, warm=start)
    return out


def predict(model: Model, theta, phase: float, design: Design, cycles: dict | None = None) -> Prediction:
    """Predictions for a design, the unperturbed cycle having started at time -phase (hours). A model that
    does not oscillate predicts flat time courses."""
    cycles = cycles if cycles is not None else cycles_for(model, theta, design)
    base: LimitCycle | None = cycles[NONE]
    levels = []
    for c in design.courses:
        t = np.asarray(c.times)
        levels.append(np.zeros(c.n) if base is None else base.at(c.species, t, phase))
    rhythm = np.array([])
    if design.rhythm:
        vals = [base.period if base else np.nan]
        for p in design.rhythm.perturbations:
            lc = cycles.get(p)
            if base is None or lc is None:
                vals += [np.nan, np.nan]
            else:
                vals += [lc.period, np.log(lc.amplitude / base.amplitude)]
        rhythm = np.array(vals)
    return Prediction(levels=levels, rhythm=rhythm, cycles=cycles)


@dataclass
class Data:
    """Observations for a design, in the same layout as a Prediction."""
    design: Design
    levels: list[np.ndarray]
    rhythm: np.ndarray

    def vector(self) -> np.ndarray:
        return np.concatenate(self.levels + [self.rhythm])


def simulate(model: Model, theta, design: Design, rng: np.random.Generator | None = None,
             phase: float = 0.0, truth: Prediction | None = None) -> Data:
    """Data from a model: its predictions plus measurement noise (rng None gives noise-free data)."""
    truth = truth if truth is not None else predict(model, theta, phase, design)
    if not truth.oscillates or (design.rhythm and np.isnan(truth.rhythm).any()):
        raise ValueError(f"{model.key} does not oscillate under every condition in {design.key}")
    levels = [lv.copy() for lv in truth.levels]
    rhythm = truth.rhythm.copy()
    if rng is not None:
        for i, c in enumerate(design.courses):
            levels[i] = levels[i] + rng.normal(0.0, c.sigma, c.n)
        if design.rhythm:
            rhythm = rhythm + rng.normal(0.0, 1.0, rhythm.size) * design.sigmas[-rhythm.size:]
    return Data(design=design, levels=levels, rhythm=rhythm)
