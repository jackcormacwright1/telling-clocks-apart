"""Three rival explanations of how a single negative-feedback loop keeps time.

In each model a clock gene's mRNA (stage 1) is translated into protein, the protein passes through one or
more forms, and the last form represses the gene's own transcription. That is the core of the mammalian
clock: PER and CRY proteins shutting down their own genes. A loop like this only oscillates, rather than
settling to a steady level, if the repression is strong and late enough. The three models differ in what
supplies that:

  switch       Goodwin's loop (1965): mRNA, protein, repressor, all removed in proportion to their amount.
               Oscillation needs very steep repression: a Hill coefficient above 8 when the removal rates are
               equal (Griffith, 1968).
  saturation   the same three stages, but each is removed by an enzyme that saturates (Michaelis-Menten), as
               in Gonze et al. (2005). Saturating removal lets the loop oscillate with much gentler repression.
  delay        a longer chain: mRNA and six protein forms (for example successive phosphorylations and
               nuclear entry). Each extra step adds delay, and with seven stages a Hill coefficient above about
               2.1 is enough (the secant condition of Tyson and Othmer, 1978).

Equations, for stages i = 1..N with x_N the repressor:

  dx_1/dt = v / (1 + x_N^n) - removal_1(x_1)
  dx_i/dt = k_i x_(i-1) - removal_i(x_i)            i = 2..N

with removal_i(x) = r_i x (switch, delay) or V_i x / (Km + x) (saturation). Concentrations are in units in
which repression is half-maximal at x_N = 1, and time is in hours. In the switch and delay models each
protein stage is made at the same rate as it is removed (k_i = r_i), which only fixes the units of the
intermediate stages, and v = gain x r_1, so "gain" is the level the loop would reach without repression.
With these conventions multiplying every rate by the same factor speeds the clock up by that factor and
leaves every waveform unchanged.

Two perturbations stand for real experiments:

  inhibitor  multiplies the removal rate of every protein stage (2..N), as a drug that slows PER
             degradation does (CK1 delta/epsilon inhibitors lengthen the period).
  dosage     multiplies the transcription rate v, as removing one copy of the gene does (0.5).

Observables are log mRNA (stage 1) and log total protein (stages 2..N), both measured relative to an
unknown scale, plus the free-running period and the oscillation's amplitude.

The three hypotheses are kept distinct by the bounds on the Hill coefficient: steep in the switch model
(4 to 60), gentle in the other two (1 to 4). With steep repression allowed, the saturation model would
contain the switch as a special case (removal far from saturation is linear), and so, roughly, would the
delay model; the question here is what supplies the nonlinearity when repression is not steep.

Each model's base parameters (found by scripts/tune_models.py) give a 24 h period and the same 47-fold
peak-to-trough swing in mRNA (3.84 on the log scale), and keep oscillating under every perturbation in the
experiments. Matching the swing takes a Hill coefficient of 20 in the switch model, 2 in the saturation
model and 3 in the delay model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

DT = 0.05        # integration step, hours
GRID = 256       # points per cycle in a LimitCycle


# ---------------------------------------------------------------- integration (numba)

@njit(cache=True)
def _rhs(x, v, n, k, V, Km, linear):
    N = x.shape[0]
    d = np.empty(N)
    rep = x[N - 1]
    for i in range(N):
        prod = v / (1.0 + rep ** n) if i == 0 else k[i] * x[i - 1]
        if linear:
            d[i] = prod - V[i] * x[i]
        else:
            d[i] = prod - V[i] * x[i] / (Km[i] + x[i])
    return d


@njit(cache=True)
def _step(x, v, n, k, V, Km, linear, dt):
    k1 = _rhs(x, v, n, k, V, Km, linear)
    k2 = _rhs(x + 0.5 * dt * k1, v, n, k, V, Km, linear)
    k3 = _rhs(x + 0.5 * dt * k2, v, n, k, V, Km, linear)
    k4 = _rhs(x + dt * k3, v, n, k, V, Km, linear)
    y = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    for i in range(y.shape[0]):
        if y[i] < 1e-12:
            y[i] = 1e-12
    return y


@njit(cache=True)
def _run(x0, v, n, k, V, Km, linear, dt, n_skip, n_rec):
    """Integrate n_skip steps without recording, then n_rec steps recording every state (fourth-order
    Runge-Kutta, fixed step)."""
    x = x0.copy()
    for s in range(n_skip):
        x = _step(x, v, n, k, V, Km, linear, dt)
    out = np.empty((n_rec + 1, x.shape[0]))
    out[0] = x
    for s in range(n_rec):
        x = _step(x, v, n, k, V, Km, linear, dt)
        out[s + 1] = x
    return out


# ---------------------------------------------------------------- models

@dataclass(frozen=True)
class Perturbation:
    inhibitor: float = 1.0   # multiplier on protein removal
    dosage: float = 1.0      # multiplier on transcription

    @property
    def label(self) -> str:
        parts = []
        if self.inhibitor != 1:
            parts.append(f"protein removal x{self.inhibitor:g}")
        if self.dosage != 1:
            parts.append(f"transcription x{self.dosage:g}")
        return ", ".join(parts) or "unperturbed"


NONE = Perturbation()


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    description: str
    n_stages: int
    linear: bool
    param_names: tuple[str, ...]
    lower: tuple[float, ...]    # bounds used when fitting (natural units)
    upper: tuple[float, ...]
    base: tuple[float, ...]     # 24 h period, log mRNA amplitude 3.84 (scripts/tune_models.py)
    param_notes: dict = field(default_factory=dict, compare=False, hash=False)

    def kinetics(self, theta, pert: Perturbation = NONE):
        """(v, n, k, V, Km) arrays for the integrator."""
        p = dict(zip(self.param_names, np.asarray(theta, dtype=float)))
        N = self.n_stages
        Km = np.ones(N)
        if self.key == "switch":
            r = np.array([p["r_mrna"], p["r_protein"], p["r_repressor"]])
            v, k, V = p["gain"] * r[0], r.copy(), r.copy()
        elif self.key == "delay":
            r = np.r_[p["r_mrna"], np.full(N - 1, p["r_protein"])]
            v, k, V = p["gain"] * r[0], r.copy(), r.copy()
        elif self.key == "saturation":
            v = p["v"]
            k = np.full(N, p["k"])
            V = np.array([p["V_mrna"], p["V_protein"], p["V_repressor"]])
            Km = np.full(N, p["Km"])
        else:
            raise ValueError(self.key)
        k[0] = 0.0
        V[1:] *= pert.inhibitor
        return v * pert.dosage, p["hill"], k, V, Km

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def log_bounds(self):
        return np.log(self.lower), np.log(self.upper)


SWITCH = Model(
    key="switch", name="Steep switch",
    description="Three stages removed in proportion to their amount; oscillates only with very steep repression.",
    n_stages=3, linear=True,
    param_names=("r_mrna", "r_protein", "r_repressor", "hill", "gain"),
    lower=(0.02, 0.02, 0.02, 4.0, 1.5), upper=(2.0, 2.0, 2.0, 60.0, 1e4),
    base=(0.2217, 0.1847, 0.1478, 20.0, 17.01),
    param_notes={"r_mrna": "mRNA removal rate (/h)", "r_protein": "protein removal rate (/h)",
                 "r_repressor": "repressor removal rate (/h)", "hill": "steepness of repression (Hill coefficient)",
                 "gain": "loop gain: the level the loop would reach without repression"},
)

SATURATION = Model(
    key="saturation", name="Saturating removal",
    description="Three stages each removed by a saturating enzyme; oscillates with gentler repression.",
    n_stages=3, linear=False,
    param_names=("v", "hill", "k", "V_mrna", "V_protein", "V_repressor", "Km"),
    lower=(0.05, 1.0, 0.02, 0.02, 0.02, 0.02, 0.02), upper=(10.0, 4.0, 5.0, 5.0, 5.0, 5.0, 20.0),
    base=(0.949, 2.0, 0.8134, 0.4067, 0.4067, 0.4067, 0.2),
    param_notes={"v": "maximal transcription rate", "hill": "steepness of repression (Hill coefficient)",
                 "k": "translation and conversion rate (/h)",
                 "V_mrna": "maximal mRNA removal rate", "V_protein": "maximal protein removal rate",
                 "V_repressor": "maximal repressor removal rate", "Km": "half-saturation level of removal"},
)

DELAY = Model(
    key="delay", name="Long delay chain",
    description="mRNA and six protein forms in sequence; the extra steps supply the delay.",
    n_stages=7, linear=True,
    param_names=("r_mrna", "r_protein", "hill", "gain"),
    lower=(0.02, 0.05, 1.0, 1.1), upper=(3.0, 5.0, 4.0, 1e4),
    base=(0.5639, 0.5639, 3.0, 15.22),
    param_notes={"r_mrna": "mRNA removal rate (/h)", "r_protein": "rate of each protein step (/h)",
                 "hill": "steepness of repression (Hill coefficient)", "gain": "loop gain"},
)

MODELS = {m.key: m for m in (SWITCH, SATURATION, DELAY)}


# ---------------------------------------------------------------- limit cycles

@dataclass
class LimitCycle:
    period: float               # hours
    amplitude: float            # peak minus trough of log mRNA
    log_mrna: np.ndarray        # GRID values over one cycle, starting where log mRNA rises through its mean
    log_protein: np.ndarray     # log total protein (stages 2..N) at the same phases
    state: np.ndarray           # full state at the start of the cycle (for warm starts)

    def at(self, species: str, t: np.ndarray, phase: float) -> np.ndarray:
        """Values at times t (hours), the cycle having started at time -phase."""
        values = self.log_mrna if species == "mrna" else self.log_protein
        u = np.mod((np.asarray(t, dtype=float) + phase) / self.period, 1.0) * GRID
        i = np.floor(u).astype(int) % GRID
        w = u - np.floor(u)
        return values[i] * (1 - w) + values[(i + 1) % GRID] * w


def limit_cycle(model: Model, theta, pert: Perturbation = NONE, warm: np.ndarray | None = None,
                max_period: float = 60.0, tol: float = 1e-3) -> LimitCycle | None:
    """The model's sustained oscillation, or None if it settles to a steady state (or its period exceeds
    max_period hours).

    Integrates until two successive cycles agree to tol (relative period) and the amplitude is no longer
    shrinking. warm is a state to start from (e.g. a nearby parameter set's cycle), which makes repeated
    calls during fitting much faster.
    """
    v, n, k, V, Km = model.kinetics(theta, pert)
    x = np.full(model.n_stages, 0.5) if warm is None else np.asarray(warm, dtype=float).copy()
    skip_hours = 600.0 if warm is None else 3 * max_period
    rec_hours = 2.5 * max_period
    n_rec = int(rec_hours / DT)
    prev_amp = None
    for attempt in range(6):
        out = _run(x, v, n, k, V, Km, model.linear, DT, int(skip_hours / DT), n_rec)
        x = out[-1]
        m = np.log(out[:, 0])
        amp = m.max() - m.min()
        if amp < 1e-3:
            return None
        mu = 0.5 * (m.max() + m.min())
        up = np.flatnonzero((m[:-1] < mu) & (m[1:] >= mu))
        if len(up) >= 3:
            tc = (up + (mu - m[up]) / (m[up + 1] - m[up])) * DT
            periods = np.diff(tc)
            p1, p2 = periods[-2], periods[-1]
            seg = m[up[-3]:up[-2] + 1]
            seg2 = m[up[-2]:up[-1] + 1]
            a1, a2 = seg.max() - seg.min(), seg2.max() - seg2.min()
            if abs(p1 - p2) / p2 < tol and a2 > 0.995 * a1:
                return _extract(model, out, tc[-2], p2, a2)
            if prev_amp is not None and a2 < 0.98 * a1 and a2 < 0.5 * prev_amp:
                return None  # amplitude decaying: a damped approach to a steady state
            prev_amp = a2
        elif len(up) < 2 and attempt >= 1:
            return None  # no oscillation within the window: steady state or period > max_period
        skip_hours = 4 * max_period
    return None


def _extract(model, out, t0, period, amplitude) -> LimitCycle:
    t = np.arange(out.shape[0]) * DT
    grid_t = t0 + period * np.arange(GRID) / GRID
    log_m = np.interp(grid_t, t, np.log(out[:, 0]))
    log_p = np.interp(grid_t, t, np.log(out[:, 1:].sum(axis=1)))
    i0 = int(t0 / DT)
    w = t0 / DT - i0
    state = out[i0] * (1 - w) + out[min(i0 + 1, len(out) - 1)] * w
    return LimitCycle(period=float(period), amplitude=float(amplitude), log_mrna=log_m, log_protein=log_p,
                      state=state)


def rescale_time(model: Model, theta, factor: float) -> np.ndarray:
    """Parameters with every rate multiplied by factor, which divides the period by factor and leaves the
    waveform unchanged."""
    p = dict(zip(model.param_names, np.asarray(theta, dtype=float)))
    rates = {"switch": ("r_mrna", "r_protein", "r_repressor"), "delay": ("r_mrna", "r_protein"),
             "saturation": ("v", "k", "V_mrna", "V_protein", "V_repressor")}[model.key]
    for name in rates:
        p[name] *= factor
    return np.array([p[name] for name in model.param_names])
