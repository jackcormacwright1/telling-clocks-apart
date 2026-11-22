"""Draw the charts in results/ from the results files (skips any whose inputs are missing).

Usage:  python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clocks.experiments import DESIGNS, INHIBITOR_DOSES  # noqa: E402
from clocks.models import MODELS, Perturbation, limit_cycle  # noqa: E402

RESULTS = ROOT / "results"
COLOURS = {"switch": "#2a78d6", "saturation": "#eb6834", "delay": "#1baf7a"}
INK, MUTED, GRID = "#1d1d1b", "#6b6a66", "#e1e0d9"
NAMES = {k: m.name for k, m in MODELS.items()}
DESIGN_NAMES = {k: d.name for k, d in DESIGNS.items()}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "lines.linewidth": 2.0,
})


def save(fig, name):
    fig.savefig(RESULTS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def models_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    t = np.linspace(0, 48, 400)
    for key, m in MODELS.items():
        lc = limit_cycle(m, m.base)
        for ax, species in zip(axes, ("mrna", "protein")):
            y = lc.at(species, t, 0.0)
            ax.plot(t, y - y.mean(), color=COLOURS[key], label=m.name)
    axes[0].set(title="mRNA", xlabel="hours", ylabel="log level (relative)")
    axes[1].set(title="Total protein", xlabel="hours")
    for ax in axes:
        ax.set_xticks(range(0, 49, 12))
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Three mechanisms tuned to the same 24-hour, 47-fold mRNA rhythm", x=0.01, ha="left", y=1.02)
    save(fig, "models.png")


def perturbations_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    doses = np.linspace(1.0, 0.65, 15)
    for key, m in MODELS.items():
        base = limit_cycle(m, m.base)
        T, A, warm = [], [], base.state
        for f in doses:
            lc = limit_cycle(m, m.base, Perturbation(inhibitor=f), warm=warm)
            T.append(np.nan if lc is None else lc.period)
            A.append(np.nan if lc is None else lc.amplitude / base.amplitude)
            warm = lc.state if lc is not None else warm
        pct = 100 * (1 - doses)
        axes[0].plot(pct, T, color=COLOURS[key], label=m.name)
        axes[1].plot(pct, A, color=COLOURS[key])
    for f in INHIBITOR_DOSES:
        for ax in axes:
            ax.axvline(100 * (1 - f), color=GRID, lw=1, zorder=0)
    axes[0].set(xlabel="protein removal slowed by (%)", ylabel="period (hours)", title="Period")
    axes[1].set(xlabel="protein removal slowed by (%)", ylabel="amplitude relative to control",
                title="Amplitude")
    axes[0].legend(fontsize=9)
    fig.suptitle("At their base parameters, the models disagree about an inhibitor", x=0.01, ha="left", y=1.02)
    save(fig, "perturbations.png")


def discrimination_figure():
    path = RESULTS / "discrimination.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    designs = list(DESIGNS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, truth in zip(axes, MODELS):
        g = d[d["truth"] == truth]
        rivals = [r for r in MODELS if r != truth]
        x = np.arange(len(designs))
        for j, rival in enumerate(rivals):
            vals = g.set_index(["rival", "design"])["delta"].reindex([(rival, k) for k in designs]).to_numpy()
            pos = x + (j - 0.5) * 0.38
            ax.bar(pos, np.maximum(vals, 1e-2), width=0.36, color=COLOURS[rival], label=f"rival: {NAMES[rival]}")
            for p_, v in zip(pos, vals):
                if v < 0.015:
                    ax.text(p_, 0.013, "0", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.axhline(10, color=INK, lw=1, ls="--")
        ax.axhline(2, color=MUTED, lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e5)
        ax.set_xticks(x, [DESIGN_NAMES[k].replace("Add ", "+ ").replace(" time course", "") for k in designs],
                      rotation=35, ha="right")
        ax.set_title(f"Truth: {NAMES[truth]}")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Δ: evidence against the rival")
    fig.suptitle("How well can the closest version of each rival mimic the truth?  "
                 "(dashed: decisive, Δ = 10; dotted: indistinguishable, Δ = 2)", x=0.01, ha="left", y=1.03)
    save(fig, "discrimination.png")


def sloppiness_figure():
    path = RESULTS / "information.json"
    if not path.exists():
        return
    info = json.loads(path.read_text())
    designs = list(DESIGNS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, key in zip(axes, MODELS):
        for i, d in enumerate(designs):
            w = np.array(info[f"{key}/{d}"]["eigenvalues"])
            sd = 1 / np.sqrt(np.maximum(w, 1e-12))
            ax.scatter(np.full(len(sd), i), sd, marker="_", s=420, color=COLOURS[key], linewidths=2)
        ax.axhline(0.1, color=INK, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 3e3)
        ax.set_xticks(range(len(designs)), [DESIGN_NAMES[k].replace("Add ", "+ ").replace(" time course", "")
                                            for k in designs], rotation=35, ha="right")
        ax.set_title(f"{MODELS[key].name} ({MODELS[key].n_params} parameters)")
    axes[0].set_ylabel("sd of each parameter combination\n(log scale: 0.1 = 10%)")
    fig.suptitle("Each tick is one combination of parameters: below the dashed line it is determined to 10%",
                 x=0.01, ha="left", y=1.03)
    save(fig, "sloppiness.png")


def identification_figure():
    path = RESULTS / "identification.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    designs = list(DESIGNS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    x = np.arange(len(designs))
    for ax, col, title in ((axes[0], "decisive", "Data rule out both rivals (SSE margin of 10 or more)"),
                           (axes[1], "identified", "BIC picks the right model")):
        for j, truth in enumerate(MODELS):
            vals = s[s["truth"] == truth].set_index("design")[col].reindex(designs).to_numpy()
            ax.bar(x + (j - 1) * 0.27, 100 * vals, width=0.25, color=COLOURS[truth],
                   label=f"truth: {NAMES[truth]}")
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks(x, [DESIGN_NAMES[k].replace("Add ", "+ ").replace(" time course", "") for k in designs],
                      rotation=35, ha="right")
    axes[0].set_ylabel("simulated data sets (%)")
    axes[0].legend(fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.22))
    save(fig, "identification.png")


def real_figures():
    path = RESULTS / "real_data.json"
    if not path.exists():
        return
    real = json.loads(path.read_text())
    curves = pd.read_csv(RESULTS / "real_fit.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    t0 = real["ct_start"]
    ax.scatter(np.array(real["data"]["t"]) + t0, real["data"]["y"], s=14, color=INK, zorder=3, label="Per2 mRNA")
    for key in MODELS:
        m = real["models"][key]
        ax.plot(curves["t"] + t0, curves[key], color=COLOURS[key],
                label=f"{NAMES[key]} (BIC {m['bic']:.1f})")
    ax.set(xlabel="circadian time (hours)", ylabel="log expression")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "real_fit.png")

    pr = pd.read_csv(RESULTS / "real_predictions.csv")
    labels = [Perturbation(inhibitor=f).label for f in INHIBITOR_DOSES]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for j, key in enumerate(MODELS):
        g = pr[pr["model"] == key]
        for i, lab in enumerate(labels):
            h = g[g["perturbation"] == lab]
            ch = (h["period"] - h["control_period"]).dropna().to_numpy()
            stops = h["period"].isna().mean()
            pos = i + (j - 1) * 0.25
            if len(ch) >= 10:
                lo, mid, hi = np.percentile(ch, [5, 50, 95])
                ax.plot([pos, pos], [lo, hi], color=COLOURS[key], lw=6, solid_capstyle="butt", alpha=0.85)
                ax.plot([pos - 0.09, pos + 0.09], [mid, mid], color="white", lw=2)
                top = hi
            else:
                top = 0
            ax.text(pos, top + 0.6, f"stops\n{100 * stops:.0f}%", ha="center", va="bottom", fontsize=7,
                    color=MUTED)
        ax.plot([], [], color=COLOURS[key], lw=6, label=NAMES[key])
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(labels)), [f"slowed {100 * (1 - f):.0f}%" for f in INHIBITOR_DOSES])
    ax.set_ylabel("change in period (hours)")
    ax.legend(fontsize=8)
    ax.set_ylim(top=max(ax.get_ylim()[1], 0) + 3)
    ax.set_title("What each model, fitted to the liver data, predicts a CK1 inhibitor will do\n(90% intervals "
                 "among parameter sets where the clock keeps running; above: how often it stops)",
                 loc="left", fontsize=10)
    save(fig, "real_predictions.png")


if __name__ == "__main__":
    models_figure()
    perturbations_figure()
    discrimination_figure()
    sloppiness_figure()
    identification_figure()
    real_figures()
