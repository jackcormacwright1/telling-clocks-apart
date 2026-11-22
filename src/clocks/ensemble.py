"""Every parameter set consistent with the data, and what each predicts.

A best fit hides the problem this project is about: with sloppy models, very different parameter sets fit
the same data almost equally well, and they can disagree about an experiment not yet done. So instead of
one fit, sample the posterior of the log parameters and the phase by adaptive Metropolis (Haario et al.
2001), with a flat prior on the log scale within each model's bounds and the Gaussian likelihood
exp(-SSE / 2), and push every sample through the prediction. The spread of the predictions is what the
data actually say about the new experiment under that model (Gutenkunst et al. 2007 call this a sloppy
model ensemble).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .experiments import Data, Design, predict
from .fit import Fit, Problem
from .models import NONE, Model, Perturbation, limit_cycle


@dataclass
class Ensemble:
    model: Model
    samples: np.ndarray      # (draws, n_params + 1): log parameters and phase (fraction of a period)
    sse: np.ndarray          # SSE of each draw
    acceptance: float

    @property
    def thetas(self) -> np.ndarray:
        return np.exp(self.samples[:, :self.model.n_params])


def sample(model: Model, data: Data, start: Fit, n_steps: int = 30000, burn: int = 10000, thin: int = 50,
           rng: np.random.Generator | None = None, init_sd: float = 0.02) -> Ensemble:
    """Adaptive Metropolis. During burn-in the proposal's shape follows the chain's covariance and its size is
    tuned towards an acceptance rate of 0.234; after burn-in the proposal is frozen, so the draws come from
    an ordinary Metropolis chain."""
    rng = rng if rng is not None else np.random.default_rng(0)
    prob = Problem(model, data)
    x = np.r_[np.log(start.theta), start.phase / start.period]
    d = x.size
    r = prob(x)
    sse = float(r @ r)
    chol = np.eye(d) * init_sd
    log_scale = np.log(2.38 / np.sqrt(d))
    history = []
    accepted = recent = 0
    draws, draw_sse = [], []
    for step in range(1, n_steps + 1):
        y = x + np.exp(log_scale) * (chol @ rng.standard_normal(d)) if step > 500 else x + chol @ rng.standard_normal(d)
        ok = False
        if np.all(y[:-1] >= prob.lower[:-1]) and np.all(y[:-1] <= prob.upper[:-1]):
            ry = prob(y)
            sse_y = float(ry @ ry)
            if np.log(rng.uniform()) < -0.5 * (sse_y - sse):
                x, sse, ok = y, sse_y, True
        if step <= burn:
            history.append(x.copy())
            recent += ok
            if step >= 500 and step % 100 == 0:
                cov = np.cov(np.array(history[len(history) // 2:]).T) + 1e-10 * np.eye(d)
                try:
                    chol = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    pass
                if step > 500:
                    log_scale += (recent / 100 - 0.234) / np.sqrt(step / 100)
                recent = 0
        else:
            accepted += ok
            if (step - burn) % thin == 0:
                draws.append(x.copy())
                draw_sse.append(sse)
    return Ensemble(model=model, samples=np.array(draws), sse=np.array(draw_sse),
                    acceptance=accepted / max(1, n_steps - burn))


def effective_size(x: np.ndarray) -> float:
    """Effective sample size of a chain of draws (Geyer's initial positive sequence)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float(n)
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1) * np.var(x))
    total = 0.0
    for k in range(0, n - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        total += pair
    return float(n / max(1.0, 2 * total - 1))


def split_rhat(chains: np.ndarray) -> float:
    """Split R-hat (Gelman et al., Bayesian Data Analysis, 3rd ed., section 11.4) for draws shaped
    (chains, draws): near 1 when the chains agree, above about 1.1 when they have not mixed."""
    chains = np.asarray(chains, dtype=float)
    half = chains.shape[1] // 2
    parts = np.concatenate([chains[:, :half], chains[:, half:2 * half]])
    n = parts.shape[1]
    w = parts.var(axis=1, ddof=1).mean()
    b = n * parts.mean(axis=1).var(ddof=1)
    if w == 0:
        return 1.0
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def predict_rhythms(ensemble: Ensemble, perturbations: tuple[Perturbation, ...]) -> np.ndarray:
    """For each draw: the control period, then the period and log amplitude ratio under each perturbation
    (NaN where the clock stops)."""
    model = ensemble.model
    out = np.full((len(ensemble.samples), 1 + 2 * len(perturbations)), np.nan)
    warm = {}
    for i, theta in enumerate(ensemble.thetas):
        base = limit_cycle(model, theta, NONE, warm=warm.get(NONE))
        if base is None:
            continue
        warm[NONE] = base.state
        out[i, 0] = base.period
        for j, p in enumerate(perturbations):
            lc = limit_cycle(model, theta, p, warm=warm.get(p, base.state))
            if lc is not None:
                warm[p] = lc.state
                out[i, 1 + 2 * j] = lc.period
                out[i, 2 + 2 * j] = np.log(lc.amplitude / base.amplitude)
    return out


def predict_courses(ensemble: Ensemble, design: Design, times: np.ndarray, species: str = "mrna"):
    """Each draw's predicted log level at the given times (offset to match the data's mean on the design's
    first time course)."""
    model = ensemble.model
    out = []
    for z in ensemble.samples:
        theta = np.exp(z[:model.n_params])
        lc = limit_cycle(model, theta)
        if lc is None:
            continue
        phase = z[-1] * lc.period
        pred = predict(model, theta, phase, design, {NONE: lc})
        out.append((lc.at(species, times, phase), pred.levels[0]))
    return out
