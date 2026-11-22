"""Which experiment tells the three clock models apart?

For each model taken as the truth and each experiment, fits both rival models to the truth's noise-free
predictions (Delta, the rival's best SSE, is the evidence the experiment gives against it), and computes the
Fisher information of the truth's own parameters. Writes results/discrimination.csv,
results/rival_fits.json and results/information.json.

A local minimum would overstate Delta, so each fit is run from many starts: the rival's base parameters,
random draws around them, and (in a second pass) the rival's best fits to every other experiment. With
--refine, it instead reruns every fit from the saved fits plus more widely spread random starts, keeping
whichever is better.

With --only, it fits just the named experiments (for example one added later), starting from the saved
fits to all the others as well, and merges them into the saved results.

Usage:  python scripts/run_design.py [--workers 2] [--random 6]
        python scripts/run_design.py --refine [--random 12]
        python scripts/run_design.py --only both [--random 12]
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

from clocks.design import information, rival_fit  # noqa: E402
from clocks.experiments import DESIGNS  # noqa: E402
from clocks.models import MODELS  # noqa: E402

RESULTS = ROOT / "results"


def refine_job(args):
    truth, rival, n_random, seed, saved = args
    t0 = time.time()
    fits = {}
    for d in DESIGNS:
        starts = [np.array(saved[d]["theta"])] + [np.array(saved[o]["theta"]) for o in DESIGNS if o != d]
        f = rival_fit(truth, rival, d, n_random=n_random, seed=seed + 1, starts=starts, spread=1.0)
        fits[d] = f
    print(f"  refined {truth} -> {rival}: {time.time() - t0:.0f}s", flush=True)
    return truth, rival, {d: _record(f) for d, f in fits.items()}


def _record(f):
    return {"delta": f.sse, "theta": f.theta.tolist(), "period": f.period, "phase": f.phase, "k": f.k,
            "n_obs": f.n_obs}


def pair_job(args):
    truth, rival, n_random, seed = args
    t0 = time.time()
    fits = {}
    starts = []
    for d in DESIGNS:  # pass 1: each experiment, also starting from the fits so far
        f = rival_fit(truth, rival, d, n_random=n_random, seed=seed, starts=list(starts))
        fits[d] = f
        starts.append(f.theta)
    for d in DESIGNS:  # pass 2: every experiment from every fit
        others = [fits[o].theta for o in DESIGNS if o != d]
        f = rival_fit(truth, rival, d, n_random=0, starts=others)
        if f.sse < fits[d].sse - 1e-9:
            fits[d] = f
    print(f"  {truth} -> {rival}: {time.time() - t0:.0f}s", flush=True)
    return truth, rival, {d: _record(f) for d, f in fits.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--random", type=int, default=6)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(DESIGNS))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    if args.refine:
        refine(args, t0)
        return
    if args.only:
        only(args, t0)
        return

    info = {}
    for m in MODELS:
        for d in DESIGNS:
            i = information(m, d)
            info[f"{m}/{d}"] = {"eigenvalues": i.eigenvalues.tolist(), "se_log": i.se_log.tolist(),
                                "eigenvectors": i.eigenvectors.tolist(), "n_determined": i.n_determined(),
                                "params": list(MODELS[m].param_names)}
    (RESULTS / "information.json").write_text(json.dumps(info, indent=1))
    print(f"information done ({time.time() - t0:.0f}s)", flush=True)

    jobs = [(t, r, args.random, 17 * i) for i, (t, r) in
            enumerate((t, r) for t in MODELS for r in MODELS if r != t)]
    with Pool(args.workers) as pool:
        out = pool.map(pair_job, jobs, chunksize=1)
    write(out, t0)


def only_job(args):
    truth, rival, designs, n_random, seed, saved = args
    t0 = time.time()
    starts = [np.array(f["theta"]) for f in saved.values()]
    out = {}
    for d in designs:
        narrow = rival_fit(truth, rival, d, n_random=n_random // 2, seed=seed, starts=starts)
        wide = rival_fit(truth, rival, d, n_random=n_random - n_random // 2, seed=seed + 1, spread=1.0)
        out[d] = _record(min(narrow, wide, key=lambda f: f.sse))
    print(f"  {truth} -> {rival}: {time.time() - t0:.0f}s", flush=True)
    return truth, rival, out


def only(args, t0):
    info = json.loads((RESULTS / "information.json").read_text())
    for m in MODELS:
        for d in args.only:
            i = information(m, d)
            info[f"{m}/{d}"] = {"eigenvalues": i.eigenvalues.tolist(), "se_log": i.se_log.tolist(),
                                "eigenvectors": i.eigenvectors.tolist(), "n_determined": i.n_determined(),
                                "params": list(MODELS[m].param_names)}
    (RESULTS / "information.json").write_text(json.dumps(info, indent=1))
    saved = json.loads((RESULTS / "rival_fits.json").read_text())
    pairs = [(t, r) for t in MODELS for r in MODELS if r != t]
    jobs = [(t, r, args.only, args.random, 2000 + 17 * i,
             {d: saved[f"{t}/{r}/{d}"] for d in DESIGNS if f"{t}/{r}/{d}" in saved})
            for i, (t, r) in enumerate(pairs)]
    with Pool(args.workers) as pool:
        out = pool.map(only_job, jobs, chunksize=1)
    merged = []
    for truth, rival, new in out:
        by_design = {d: saved[f"{truth}/{rival}/{d}"] for d in DESIGNS if f"{truth}/{rival}/{d}" in saved}
        by_design.update(new)
        merged.append((truth, rival, by_design))
    write(merged, t0)


def refine(args, t0):
    saved = json.loads((RESULTS / "rival_fits.json").read_text())
    pairs = [(t, r) for t in MODELS for r in MODELS if r != t]
    jobs = [(t, r, args.random, 1000 + 17 * i, {d: saved[f"{t}/{r}/{d}"] for d in DESIGNS})
            for i, (t, r) in enumerate(pairs)]
    with Pool(args.workers) as pool:
        out = pool.map(refine_job, jobs, chunksize=1)
    merged = []
    for truth, rival, by_design in out:
        best = {}
        for d, f in by_design.items():
            old = saved[f"{truth}/{rival}/{d}"]
            if f["delta"] < old["delta"] - 1e-9:
                print(f"  {truth} -> {rival} {d}: {old['delta']:.2f} -> {f['delta']:.2f}")
                best[d] = f
            else:
                best[d] = old
        merged.append((truth, rival, best))
    write(merged, t0)


def write(out, t0):
    fits, rows = {}, []
    for truth, rival, by_design in out:
        for d, f in by_design.items():
            fits[f"{truth}/{rival}/{d}"] = f
            rows.append({"truth": truth, "rival": rival, "design": d, "delta": f["delta"]})
    (RESULTS / "rival_fits.json").write_text(json.dumps(fits, indent=1))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "discrimination.csv", index=False)
    wide = table.pivot_table(index=["truth", "rival"], columns="design", values="delta")[list(DESIGNS)]
    print(wide.round(1).to_string())
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
