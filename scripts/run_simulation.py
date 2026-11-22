"""How often does each experiment identify the right model from noisy data?

For each model as the truth and each experiment: simulate data with measurement noise (and a random phase)
and fit all three models. Two summaries per experiment:

  decisive    the truth fits better than every rival by at least 10 in SSE (a likelihood ratio of about
              150), i.e. the data themselves rule the rivals out. This is the noisy counterpart of Delta.
  bic         the model with the lowest BIC is the truth. BIC also rewards fewer parameters, so when all three
              fit equally it picks the simplest (the delay model), right or wrong.

Each fit starts from that model's best fit to the truth's noise-free data (results/rival_fits.json, from
run_design.py; for the true model, its own parameters) and from its base parameters. Writes results/simulation.csv (one row per data set and model)
and results/identification.csv (the summary).

Usage:  python scripts/run_simulation.py [--reps 20] [--workers 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clocks.experiments import DESIGNS, simulate  # noqa: E402
from clocks.fit import fit  # noqa: E402
from clocks.models import MODELS  # noqa: E402

RESULTS = ROOT / "results"
RIVALS = json.loads((RESULTS / "rival_fits.json").read_text()) if (RESULTS / "rival_fits.json").exists() else {}


def job(args):
    truth, design, rep, seed = args
    rng = np.random.default_rng(seed)
    t = MODELS[truth]
    data = simulate(t, t.base, DESIGNS[design], rng=rng, phase=float(rng.uniform(0, 24)))
    rows = []
    for key, m in MODELS.items():
        start = t.base if key == truth else RIVALS[f"{truth}/{key}/{design}"]["theta"]
        f = fit(m, data, starts=[start], n_random=0, rng=rng)
        rows.append({"truth": truth, "design": design, "rep": rep, "model": key, "sse": f.sse, "k": f.k,
                     "n_obs": f.n_obs, "bic": f.bic})
    return rows


DECISIVE = 10.0


def summarise(sim: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (truth, design), g in sim.groupby(["truth", "design"]):
        bic = g.pivot(index="rep", columns="model", values="bic")
        sse = g.pivot(index="rep", columns="model", values="sse")
        rivals = [m for m in sse.columns if m != truth]
        margin = sse[rivals].min(axis=1) - sse[truth]
        chosen = bic.idxmin(axis=1)
        rel = bic.sub(bic.min(axis=1), axis=0)
        weight = np.exp(-0.5 * rel)
        post = weight.div(weight.sum(axis=1), axis=0)[truth]
        out.append({"truth": truth, "design": design, "reps": len(bic),
                    "decisive": float((margin >= DECISIVE).mean()),
                    "misled": float((margin <= -DECISIVE).mean()),
                    "margin_median": float(margin.median()),
                    "identified": float((chosen == truth).mean()),
                    "posterior_truth": float(post.mean()),
                    "chosen": "; ".join(f"{k} {v}" for k, v in chosen.value_counts().items())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if not RIVALS:
        sys.exit("run scripts/run_design.py first")
    t0 = time.time()
    jobs = [(t, d, r, 1000 * i + r) for i, (t, d) in enumerate((t, d) for t in MODELS for d in DESIGNS)
            for r in range(args.reps)]
    rows = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
            rows += res
            if i % 20 == 0 or i == len(jobs):
                pd.DataFrame(rows).to_csv(RESULTS / "simulation.csv", index=False)
                print(f"  {i}/{len(jobs)} data sets ({time.time() - t0:.0f}s)", flush=True)
    sim = pd.DataFrame(rows).sort_values(["truth", "design", "rep", "model"])
    sim.to_csv(RESULTS / "simulation.csv", index=False)
    summary = summarise(sim)
    summary.to_csv(RESULTS / "identification.csv", index=False)
    for col in ("decisive", "identified"):
        print(col)
        print(summary.pivot(index="truth", columns="design", values=col)[list(DESIGNS)].round(2))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
