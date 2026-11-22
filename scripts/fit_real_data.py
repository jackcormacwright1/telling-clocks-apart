"""Fit the three models to real data: Per2 mRNA in mouse liver, every hour for two days.

Hughes et al. (2009) sampled the livers of mice kept in constant darkness every hour from CT18 to CT65
(GEO GSE11923, Affymetrix Mouse Genome 430 2.0). The series matrix holds expression on a linear scale, so
it is logged here. Per2 has two probe sets on the array (MGI:1195265): 1417602_at and 1417603_at; the
script uses the more strongly expressed. This script fits each model
to the Per2 time course, compares them by BIC, samples each model's posterior, and predicts what each says
the inhibitor series and half dosage would do. Writes results/real_data.json, results/real_fit.csv and
results/real_predictions.csv.

The noise level is estimated from the residuals of a harmonic regression (24, 12 and 8 hour components and
a linear trend), which describes a smooth rhythm without assuming any of the three models.

Usage:
  python scripts/fit_real_data.py [--matrix data/GSE11923_series_matrix.txt.gz]
  python scripts/fit_real_data.py --annotation <chip or GEO platform file>   (find the probe sets by symbol)
  python scripts/fit_real_data.py --probe 1417602_at                          (choose the probe set)
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clocks.ensemble import effective_size, predict_rhythms, sample, split_rhat  # noqa: E402
from clocks.experiments import PERTURBATIONS, Data, Design, TimeCourse  # noqa: E402
from clocks.fit import fit  # noqa: E402
from clocks.models import MODELS, limit_cycle  # noqa: E402

RESULTS = ROOT / "results"
GENE = "Per2"
PROBES = ["1417602_at", "1417603_at"]  # Per2 on the Mouse Genome 430 2.0 array, from MGI:1195265


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """The expression table (probe sets x samples) and each sample's circadian time."""
    titles, rows, header, in_table = [], [], None, False
    characteristics = []
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                characteristics.append([x.strip('"') for x in line.split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                parts = [x.strip('"') for x in line.split("\t")]
                if header is None:
                    header = parts
                else:
                    rows.append(parts)
    table = pd.DataFrame(rows, columns=header).set_index("ID_REF").apply(pd.to_numeric, errors="coerce")
    ct = []
    for j, title in enumerate(titles):
        texts = [title] + [c[j] for c in characteristics if j < len(c)]
        hit = next((m for t in texts for m in [re.search(r"CT\s*(\d+)", t, re.I)] if m), None)
        if hit is None:
            raise ValueError(f"no circadian time in sample {title!r}")
        ct.append(int(hit.group(1)))
    return table, pd.Series(ct, index=table.columns, name="ct")


def probes_for(gene: str, annotation: Path) -> list[str]:
    """Probe sets annotated to the gene, from a chip file (MSigDB) or a GEO platform annotation."""
    with _open(annotation) as f:
        lines = [ln for ln in f if not ln.startswith(("#", "!", "^"))]
    from io import StringIO
    ann = pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str)
    id_col = next(c for c in ann.columns if c.lower().replace(" ", "_") in ("probe_set_id", "id"))
    sym_col = next(c for c in ann.columns if c.lower().replace(" ", "_") in ("gene_symbol", "identifier"))
    return ann.loc[ann[sym_col].fillna("").str.split(r"\s*///\s*").map(lambda s: gene in s), id_col].tolist()


def noise_sd(t: np.ndarray, y: np.ndarray) -> float:
    cols = [np.ones_like(t), t - t.mean()]
    for T in (24.0, 12.0, 8.0):
        cols += [np.cos(2 * np.pi * t / T), np.sin(2 * np.pi * t / T)]
    X = np.column_stack(cols)
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))


def summarise_prediction(h: pd.DataFrame) -> dict:
    """How often the clock stops, and among the draws where it keeps going, the change in period."""
    change = (h["period"] - h["control_period"]).dropna()
    out = {"stops": float(h["period"].isna().mean()), "running": int(len(change))}
    if len(change):
        out.update({"period_change_median": float(change.median()),
                    "period_change_90": [float(change.quantile(0.05)), float(change.quantile(0.95))],
                    "lengthens": float((change > 0).mean()),
                    "amplitude_ratio_median": float(np.exp(h["log_amplitude_ratio"].median()))})
    return out


def chain_job(args):
    key, data, start, steps, chain = args
    burn = steps // 3
    ens = sample(MODELS[key], data, start, n_steps=steps, burn=burn, thin=max(1, (steps - burn) // 250),
                 rng=np.random.default_rng(100 + chain))
    return key, ens.samples, predict_rhythms(ens, PERTURBATIONS), ens.acceptance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=ROOT / "data" / "GSE11923_series_matrix.txt.gz")
    ap.add_argument("--annotation", type=Path)
    ap.add_argument("--probe", nargs="+")
    ap.add_argument("--steps", type=int, default=60000, help="per chain; the first third is burn-in")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--random", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    table, ct = read_series_matrix(args.matrix)
    if args.probe:
        probes = args.probe
    elif args.annotation:
        probes = [p for p in probes_for(GENE, args.annotation) if p in table.index]
    else:
        probes = [p for p in PROBES if p in table.index]
    if not probes:
        sys.exit(f"no {GENE} probe sets found")
    order = ct.sort_values().index
    values = table.loc[probes, order]
    probe = values.mean(axis=1).idxmax()  # the most strongly expressed probe set
    raw = values.loc[probe].to_numpy(dtype=float)
    linear = np.nanmax(table.to_numpy()) > 100     # log2 values never exceed about 16
    y = np.log(raw) if linear else raw * np.log(2.0)   # natural log either way
    print(f"{GENE} probe sets {probes}; using {probe} ({'linear' if linear else 'log2'} scale, "
          f"{np.exp(y.max() - y.min()):.0f}-fold range)")
    t = (ct[order] - ct.min()).to_numpy(dtype=float)
    sigma = noise_sd(t, y)
    print(f"{len(y)} samples, CT{ct.min()} to CT{ct.max()}; noise sd {sigma:.3f} (natural log)")

    design = Design("liver", "Per2 in mouse liver", "Hughes et al. (2009)",
                    (TimeCourse("mrna", tuple(t), sigma),))
    data = Data(design=design, levels=[y], rhythm=np.array([]))
    fits = {}
    for key, m in MODELS.items():
        f = fit(m, data, n_random=args.random, rng=np.random.default_rng(7))
        fits[key] = f
        print(f"  {key:10s} SSE {f.sse:7.2f}  BIC {f.bic:7.2f}  period {f.period:5.2f} h  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # posterior samples: several chains per model from the best fit, in parallel
    jobs = [(key, data, fits[key], args.steps, c) for key in MODELS for c in range(args.chains)]
    with Pool(args.workers) as pool:
        results = pool.map(chain_job, jobs, chunksize=1)
    ensembles, preds, diagnostics = {}, {}, {}
    for key in MODELS:
        runs = [r for r in results if r[0] == key]
        samples = np.stack([r[1] for r in runs])          # (chains, draws, params)
        P = np.stack([r[2] for r in runs])                # (chains, draws, predictions)
        preds[key] = P.reshape(-1, P.shape[-1])
        ensembles[key] = samples.reshape(-1, samples.shape[-1])
        change = P[:, :, 3] - P[:, :, 0]                  # period change, protein removal x0.8
        stopped = np.isnan(change)
        diagnostics[key] = {
            "chains": len(runs), "draws": int(samples.shape[0] * samples.shape[1]),
            "acceptance": float(np.mean([r[3] for r in runs])),
            "rhat_max": max(split_rhat(samples[:, :, j]) for j in range(samples.shape[2])),
            "ess_min": float(min(sum(effective_size(samples[c, :, j]) for c in range(len(runs)))
                                 for j in range(samples.shape[2]))),
            "ess_inhibitor": float(sum(effective_size(np.where(stopped[c], 1e3, change[c]))
                                       for c in range(len(runs)))),
        }
        d = diagnostics[key]
        print(f"  {key:10s} {d['chains']} chains, {d['draws']} draws, acceptance {d['acceptance']:.2f}, "
              f"R-hat at most {d['rhat_max']:.2f}, effective sample size {d['ess_min']:.0f} (parameters), "
              f"{d['ess_inhibitor']:.0f} (inhibitor prediction)", flush=True)

    bic = pd.Series({k: f.bic for k, f in fits.items()})
    w = np.exp(-0.5 * (bic - bic.min()))
    post = (w / w.sum()).to_dict()

    grid = np.linspace(0, t.max(), 400)
    curves = {"t": grid}
    for key, f in fits.items():
        lc = limit_cycle(MODELS[key], f.theta)
        fitted = lc.at("mrna", t, f.phase)
        offset = float(np.mean(y - fitted))
        curves[key] = lc.at("mrna", grid, f.phase) + offset
    pd.DataFrame(curves).to_csv(RESULTS / "real_fit.csv", index=False)

    rows = []
    for key, P in preds.items():
        for j, p in enumerate(PERTURBATIONS):
            for i in range(len(P)):
                rows.append({"model": key, "draw": i, "perturbation": p.label, "control_period": P[i, 0],
                             "period": P[i, 1 + 2 * j], "log_amplitude_ratio": P[i, 2 + 2 * j]})
    pr = pd.DataFrame(rows)
    pr.to_csv(RESULTS / "real_predictions.csv", index=False)

    summary = {}
    for key in MODELS:
        g = pr[pr["model"] == key]
        s = {}
        for p in PERTURBATIONS:
            h = g[g["perturbation"] == p.label]
            s[p.label] = summarise_prediction(h)
        f = fits[key]
        summary[key] = {"sse": f.sse, "bic": f.bic, "k": f.k, "posterior": post[key], "period": f.period,
                        "params": {n: float(v) for n, v in f.params().items()},
                        "amplitude": limit_cycle(MODELS[key], f.theta).amplitude,
                        "sampling": diagnostics[key], "predictions": s}
    out = {"gene": GENE, "probe": probe, "probes": probes, "n": int(len(y)), "sigma": sigma,
           "ct_start": int(ct.min()), "data": {"t": t.tolist(), "y": y.tolist()}, "models": summary}
    (RESULTS / "real_data.json").write_text(json.dumps(out, indent=1))
    for key, s in summary.items():
        inh = s["predictions"][PERTURBATIONS[1].label]
        print(f"{key:10s} posterior {s['posterior']:.2f}; inhibitor x0.8 changes the period by "
              f"{inh['period_change_median']:+.1f} h (90% {inh['period_change_90'][0]:+.1f} to "
              f"{inh['period_change_90'][1]:+.1f}), stops it in {100 * inh['stops']:.0f}% of draws")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
