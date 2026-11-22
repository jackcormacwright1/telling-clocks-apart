"""Which experiment tells the models apart, and which parameters each experiment pins down.

Discrimination (T-optimality, Atkinson and Fedorov 1975). Take one model as the truth and compute its
noise-free predictions for an experiment. Fit each rival model to them. The rival's best SSE, Delta, is
how badly the closest version of the rival misses the truth, in units of measurement sd squared. With real
noise the rival's SSE exceeds the truth's by about Delta on average, so Delta is the expected evidence the
experiment provides against that rival: below about 2 the models are indistinguishable, and above about 10
the rival is decisively rejected. Because the rival is refitted, Delta measures what no setting of the
rival's parameters can mimic, not just how different the base parameter sets are. Delta grows in
proportion to replication: an experiment done twice gives twice the Delta.

Identifiability (Fisher information). Within one model, the Fisher information of the log parameters at
the truth says how precisely each combination of parameters is determined. Its eigenvalues are typically
spread over many orders of magnitude ("sloppiness", Gutenkunst et al. 2007): a few stiff combinations are
well determined and the rest are not determined at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .experiments import DESIGNS, simulate
from .fit import Fit, fisher_information, fit
from .models import MODELS

PHASE = 2.0  # hours: where in its cycle the truth is at the first sample (arbitrary)


def rival_fit(truth: str, rival: str, design: str, n_random: int = 6, seed: int = 0,
              starts: list | None = None, spread: float = 0.5) -> Fit:
    """The rival's best fit to the truth's noise-free data. Its SSE is Delta."""
    t = MODELS[truth]
    data = simulate(t, t.base, DESIGNS[design], phase=PHASE)
    return fit(MODELS[rival], data, starts=starts, n_random=n_random, rng=np.random.default_rng(seed),
               spread=spread)


@dataclass
class Information:
    model: str
    design: str
    eigenvalues: np.ndarray     # descending
    eigenvectors: np.ndarray    # columns, in the model's log parameters
    se_log: np.ndarray          # standard error of each log parameter (inf where not determined)

    def n_determined(self, threshold: float = 1.0 / 0.1 ** 2) -> int:
        """Parameter combinations determined to within 10% (information above 100)."""
        return int(np.sum(self.eigenvalues > threshold))


def information(model: str, design: str) -> Information:
    m = MODELS[model]
    data = simulate(m, m.base, DESIGNS[design], phase=PHASE)
    F = fisher_information(m, m.base, data, phase=PHASE)
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0.0, None), V[:, order]
    cond = w > w.max() * 1e-12
    cov = (V[:, cond] / w[cond]) @ V[:, cond].T
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if not cond.all():
        # a direction with no information makes every parameter it involves undetermined
        null = V[:, ~cond]
        se = np.where(np.abs(null).max(axis=1) > 1e-3, np.inf, se)
    return Information(model=model, design=design, eigenvalues=w, eigenvectors=V, se_log=se)
