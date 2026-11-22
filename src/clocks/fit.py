"""Fitting a model to data by weighted least squares, and the Fisher information of a design.

Measurement errors are Gaussian with known sds, so minus twice the log likelihood is the sum of squared
standardised residuals (SSE) plus a constant, and the best fit is a least-squares fit. Parameters are fitted
on the log scale within each model's bounds, together with the phase of the clock at the first sample.
Each time course's unknown offset is profiled out: for any parameters, the best offset is the mean
difference between data and prediction.

A model that does not oscillate where the data show a rhythm predicts flat time courses, and each rhythm
measurement it cannot predict (a period, an amplitude ratio) adds a fixed penalty of PENALTY standard
deviations. This keeps the search away from parameters that stop the clock without making the objective
undefined there.

The objective has many local minima, so each fit runs from several starting points: the starting points
given (for example a fit to similar data), plus points drawn around the model's base parameters, each
rescaled in time to match the period seen in the data and given the best phase on a grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .experiments import Data, Design, cycles_for, predict
from .models import NONE, Model, limit_cycle, rescale_time

PENALTY = 20.0
STEP = 1e-3            # finite-difference step (log parameters; phase as a fraction of the period)


class Problem:
    """Standardised residuals of one model against one data set, as a function of
    x = (log parameters..., phase as a fraction of the period)."""

    def __init__(self, model: Model, data: Data):
        self.model, self.data, self.design = model, data, data.design
        self.p = model.n_params
        lo, hi = model.log_bounds()
        self.lower = np.r_[lo, -np.inf]
        self.upper = np.r_[hi, np.inf]
        self.warm: dict = {}
        self.nfev = 0
        self._cache: tuple[bytes, np.ndarray] | None = None
        self._sig_rhythm = self.design.sigmas[-data.rhythm.size:] if data.rhythm.size else np.array([])

    def cycles(self, theta):
        cyc = cycles_for(self.model, theta, self.design, warm=self.warm)
        for k, lc in cyc.items():
            if lc is not None:
                self.warm[k] = lc.state
        return cyc

    def _residuals(self, theta, u, cycles) -> np.ndarray:
        base = cycles[NONE]
        phase = u * base.period if base is not None else 0.0
        pred = predict(self.model, theta, phase, self.design, cycles)
        out = []
        for c, y, m in zip(self.design.courses, self.data.levels, pred.levels):
            d = y - m
            out.append((d - d.mean()) / c.sigma)
        if self.data.rhythm.size:
            r = (self.data.rhythm - pred.rhythm) / self._sig_rhythm
            r[np.isnan(r)] = PENALTY
            out.append(r)
        return np.concatenate(out)

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        theta = np.exp(x[:self.p])
        r = self._residuals(theta, x[self.p], self.cycles(theta))
        self.nfev += 1
        self._cache = (key, r)
        return r

    def jac(self, x) -> np.ndarray:
        """Forward differences, stepping inwards at an upper bound."""
        x = np.asarray(x, dtype=float)
        f0 = self(x)
        J = np.empty((f0.size, x.size))
        for j in range(x.size):
            h = STEP if x[j] + STEP <= self.upper[j] else -STEP
            xj = x.copy()
            xj[j] += h
            J[:, j] = (self(xj) - f0) / h
        self._cache = (x.tobytes(), f0)
        return J

    def best_phase(self, theta, n_grid: int = 48) -> float | None:
        """The phase (fraction of a period) on a grid that best fits the time courses."""
        cycles = self.cycles(theta)
        if cycles[NONE] is None:
            return None
        grid = np.arange(n_grid) / n_grid
        sse = [np.sum(self._residuals(theta, u, cycles) ** 2) for u in grid]
        return float(grid[int(np.argmin(sse))])


@dataclass
class Fit:
    model: Model
    design: Design
    theta: np.ndarray
    phase: float           # hours: the cycle started at time -phase
    period: float          # unperturbed period at the fit (NaN if it does not oscillate)
    sse: float             # sum of squared standardised residuals = -2 log likelihood + constant
    residuals: np.ndarray
    n_starts: int
    nfev: int

    @property
    def n_obs(self) -> int:
        return self.design.n_obs

    @property
    def k(self) -> int:
        """Free parameters: the model's, the phase, and one offset per time course."""
        return self.model.n_params + 1 + len(self.design.courses)

    @property
    def bic(self) -> float:
        return self.sse + self.k * np.log(self.n_obs)

    def params(self) -> dict:
        return dict(zip(self.model.param_names, self.theta))


def estimate_period(data: Data) -> float:
    """The control period if measured; otherwise the period (18 to 32 h) of the best-fitting cosine and its
    first harmonic on the first time course."""
    if data.rhythm.size:
        return float(data.rhythm[0])
    c, y = data.design.courses[0], data.levels[0]
    t = np.asarray(c.times)
    best, best_T = np.inf, 24.0
    for T in np.arange(18.0, 32.01, 0.25):
        w = 2 * np.pi * t / T
        X = np.c_[np.ones_like(t), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)]
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        if resid @ resid < best:
            best, best_T = resid @ resid, T
    return float(best_T)


def random_starts(model: Model, n: int, target_period: float, rng: np.random.Generator,
                  spread: float = 0.5, max_tries: int = 40) -> list[np.ndarray]:
    """Parameter sets drawn log-normally around the model's base (sd spread on the log scale), kept if they
    oscillate, and rescaled in time to the target period."""
    lo, hi = model.log_bounds()
    out = []
    for _ in range(max_tries):
        if len(out) >= n:
            break
        z = np.clip(np.log(model.base) + rng.normal(0.0, spread, model.n_params), lo, hi)
        lc = limit_cycle(model, np.exp(z))
        if lc is None:
            continue
        theta = rescale_time(model, np.exp(z), lc.period / target_period)
        if np.all(np.log(theta) >= lo) and np.all(np.log(theta) <= hi):
            out.append(theta)
    return out


def fit(model: Model, data: Data, starts: list | None = None, n_random: int = 4,
        rng: np.random.Generator | None = None, max_nfev: int = 120, include_base: bool = True,
        spread: float = 0.5) -> Fit:
    """Least-squares fit from several starting points; returns the best."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = estimate_period(data)
    candidates = [np.asarray(s, dtype=float) for s in (starts or [])]
    if include_base:
        lc = limit_cycle(model, model.base)
        candidates.append(rescale_time(model, model.base, lc.period / T))
    candidates += random_starts(model, n_random, T, rng, spread=spread)
    lo, hi = model.log_bounds()
    best = None
    total_nfev = 0
    for theta0 in candidates:
        prob = Problem(model, data)
        theta0 = np.exp(np.clip(np.log(theta0), lo, hi))
        u0 = prob.best_phase(theta0)
        if u0 is None:
            continue
        x0 = np.r_[np.log(theta0), u0]
        try:
            res = least_squares(prob, x0, jac=prob.jac, bounds=(prob.lower, prob.upper), method="trf",
                                x_scale=1.0, max_nfev=max_nfev, xtol=1e-6, ftol=1e-8)
        except (ValueError, FloatingPointError):
            continue
        total_nfev += prob.nfev
        sse = float(res.fun @ res.fun)
        if best is None or sse < best[0]:
            best = (sse, res.x, res.fun, prob)
    if best is None:
        raise RuntimeError(f"no starting point for {model.key} oscillates")
    sse, x, fun, prob = best
    theta = np.exp(x[:model.n_params])
    base = prob.cycles(theta)[NONE]
    period = base.period if base is not None else np.nan
    phase = float(np.mod(x[-1], 1.0) * period) if base is not None else 0.0
    return Fit(model=model, design=data.design, theta=theta, phase=phase, period=period, sse=sse,
               residuals=fun, n_starts=len(candidates), nfev=total_nfev)


def fisher_information(model: Model, theta, data: Data, phase: float = 0.0) -> np.ndarray:
    """Fisher information for the model's log parameters at theta, with the phase (and the offsets)
    profiled out: J'J from the standardised residuals, then the Schur complement removing the phase."""
    prob = Problem(model, data)
    lc = prob.cycles(np.asarray(theta, dtype=float))[NONE]
    x = np.r_[np.log(theta), phase / lc.period]
    J = prob.jac(x)
    F = J.T @ J
    p = model.n_params
    Ftt, Ftu, Fuu = F[:p, :p], F[:p, p:], F[p:, p:]
    return Ftt - Ftu @ np.linalg.solve(Fuu, Ftu.T)
