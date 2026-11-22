"""Telling clocks apart: a Streamlit dashboard.

Run locally:   streamlit run app.py
It reads the results written by the scripts in scripts/ and runs the models live in "Explore a model".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from clocks.experiments import DESIGNS, INHIBITOR_DOSES  # noqa: E402
from clocks.models import MODELS, NONE, Perturbation, limit_cycle  # noqa: E402

st.set_page_config(page_title="Telling clocks apart", page_icon="⏱️", layout="wide")

RESULTS = ROOT / "results"
METHODOLOGY_FILE = ROOT / "docs" / "METHODOLOGY.md"
COLOURS = {"switch": "#2a78d6", "saturation": "#eb6834", "delay": "#1baf7a"}
NAMES = {k: m.name for k, m in MODELS.items()}
DESIGN_NAMES = {k: d.name for k, d in DESIGNS.items()}
NEUTRAL, MUTED, GRID, BAND, INK = "#52514e", "#898781", "#e1e0d9", "#f0efea", "#1d1d1b"
MODEL_SCALE = alt.Scale(domain=[NAMES[k] for k in MODELS], range=[COLOURS[k] for k in MODELS])

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}
.eyebrow {{ color: {NEUTRAL}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            margin: 0.4rem 0 0.35rem; }}
.lede {{ color: #3a3936; font-size: 1.02rem; margin: -0.4rem 0 0.6rem; max-width: 62rem; }}
.kpi {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.8rem 1rem 0.85rem; height: 100%; }}
.kpi .label {{ color: {NEUTRAL}; font-size: 0.86rem; line-height: 1.3; min-height: 2.2rem; }}
.kpi .value {{ color: {INK}; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0.15rem 0 0.3rem;
               font-variant-numeric: tabular-nums; white-space: nowrap; }}
.kpi .note {{ color: {MUTED}; font-size: 0.78rem; line-height: 1.35; }}
@media (max-width: 1100px) {{ .kpi .value {{ font-size: 1.45rem; }} }}
.callout {{ background: #ffffff; border: 1px solid {GRID}; border-left: 4px solid #2a78d6; border-radius: 8px;
            padding: 0.85rem 1.1rem; margin: 0.3rem 0 1rem; }}
.chart-note {{ color: {MUTED}; font-size: 0.8rem; margin: -0.3rem 0 0.4rem; }}
</style>
""", unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def legend(items: list[tuple[str, str]]) -> None:
    chips = "".join(f'<span style="margin-right:1.2rem;white-space:nowrap"><span style="display:inline-block;'
                    f'width:12px;height:3px;background:{c};vertical-align:middle;margin-right:0.4rem"></span>'
                    f'{name}</span>' for name, c in items)
    st.markdown(f'<div style="color:{NEUTRAL};font-size:0.86rem;margin:0.1rem 0 0.2rem">{chips}</div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str, text: str) -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="note">{text}</div></div>')


def short(design: str) -> str:
    return DESIGN_NAMES[design].replace("Add ", "+ ").replace(" time course", "")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def cycle(key: str, theta: tuple, inhibitor: float = 1.0, dosage: float = 1.0):
    lc = limit_cycle(MODELS[key], np.array(theta), Perturbation(inhibitor=inhibitor, dosage=dosage))
    if lc is None:
        return None
    return {"period": lc.period, "amplitude": lc.amplitude, "mrna": lc.log_mrna, "protein": lc.log_protein}


def waveform(c: dict, species: str, hours: float = 48.0) -> tuple[np.ndarray, np.ndarray]:
    values = c[species]
    n = len(values)
    t = np.linspace(0, hours, 300)
    u = np.mod(t / c["period"], 1.0) * n
    i = np.floor(u).astype(int) % n
    w = u - np.floor(u)
    y = values[i] * (1 - w) + values[(i + 1) % n] * w
    return t, y - values.mean()


disc = load_csv("discrimination.csv")
ident = load_csv("identification.csv")
info = load_json("information.json")
real = load_json("real_data.json")

# ---------------------------------------------------------------- header
eyebrow("Mathematical biology · identifiability · experimental design")
st.title("Telling clocks apart")
st.markdown('<p class="lede">Three textbook explanations of how one gene keeps 24-hour time can be tuned to '
            "produce the same daily rhythm. Which experiment would tell them apart, and how much can a time "
            "course really say about the parameters inside a model?</p>", unsafe_allow_html=True)

if disc is not None:
    worst = disc.groupby("design")["delta"].min()
    singles = worst[["protein", "inhibitor", "dosage"]]
    best_single = singles.idxmax()
    cols = st.columns(4)
    cards = [
        kpi("Rival models", "3", "A steep switch, saturating removal and a long delay chain: the three standard "
            "ways a negative-feedback loop can oscillate."),
        kpi("Standard time course, worst case", f"Δ = {worst['standard']:.1f}",
            "mRNA every 4 hours for two days. If the delay chain is true, both rivals can be tuned to mimic it "
            "exactly. Hourly sampling barely helps."),
        kpi("Best single extra experiment, worst case", f"Δ = {singles.max():.0f}",
            f"{DESIGN_NAMES[best_single].replace('Add ', 'Adding ')}. Below 10, not decisive: whichever single "
            "experiment is added, some truth leaves a rival able to fake it."),
        kpi("Both perturbations together, worst case", f"Δ = {worst['both']:.0f}",
            "An inhibitor series and half dosage, no new time courses. Each alone can be faked; together they "
            "rule out every rival whichever model is true."),
    ]
    for c, card in zip(cols, cards):
        c.markdown(card, unsafe_allow_html=True)
    st.write("")

tabs = st.tabs(["The models", "Explore a model", "Which experiment?", "What the data pin down", "Real data",
                "Methodology"])

# ---------------------------------------------------------------- the models
with tabs[0]:
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        eyebrow("One loop, three mechanisms")
        st.markdown(
            "In the mammalian clock, PER and CRY proteins shut down their own genes. A loop like that only "
            "keeps oscillating, rather than settling to a steady level, if the repression is strong and late "
            "enough. Three classic models differ in what supplies that:")
        for key, m in MODELS.items():
            st.markdown(f"<span style='color:{COLOURS[key]};font-weight:600'>■ {m.name}.</span> "
                        f"{m.description}", unsafe_allow_html=True)
        st.markdown("Each has been tuned to a 24-hour period and the same 47-fold daily swing in mRNA. That "
                    "matching took a Hill coefficient (the steepness of repression) of 20 in the switch, 2 with "
                    "saturating removal and 3 with the delay chain.")
    with right:
        rows = []
        for key, m in MODELS.items():
            c = cycle(key, tuple(m.base))
            for species, label in (("mrna", "mRNA"), ("protein", "Total protein")):
                t, y = waveform(c, species)
                rows += [{"hours": a, "level": b, "model": NAMES[key], "species": label} for a, b in zip(t, y)]
        df = pd.DataFrame(rows)
        legend([(NAMES[k], COLOURS[k]) for k in MODELS])
        for species in ("mRNA", "Total protein"):
            eyebrow(species)
            chart = alt.Chart(df[df["species"] == species]).mark_line(strokeWidth=2).encode(
                x=alt.X("hours:Q", scale=alt.Scale(domain=[0, 48]), axis=alt.Axis(values=list(range(0, 49, 12)))),
                y=alt.Y("level:Q", title="log level (relative)"),
                color=alt.Color("model:N", scale=MODEL_SCALE, legend=None),
                tooltip=["model", alt.Tooltip("hours:Q", format=".1f"), alt.Tooltip("level:Q", format=".2f")],
            ).properties(height=210)
            st.altair_chart(chart, width="stretch")
        note("The mRNA rhythms differ in shape, yet measured every 4 hours with 15% error, the delay chain's can "
             "be mimicked by a re-tuned version of either rival, and the switch's by the saturating model (see "
             "Which experiment?).")

    eyebrow("Where they part company")
    rows = []
    doses = np.linspace(1.0, 0.65, 15)
    for key, m in MODELS.items():
        base = cycle(key, tuple(m.base))
        for f in doses:
            c = cycle(key, tuple(m.base), inhibitor=float(round(f, 4)))
            if c is not None:
                rows.append({"slowed": 100 * (1 - f), "period": c["period"],
                             "amplitude": c["amplitude"] / base["amplitude"], "model": NAMES[key]})
    df = pd.DataFrame(rows)
    a = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("slowed:Q", title="protein removal slowed by (%)"),
        color=alt.Color("model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title=None)))
    c1, c2 = st.columns(2)
    c1.altair_chart(a.encode(y=alt.Y("period:Q", title="period (hours)", scale=alt.Scale(zero=False)),
                             tooltip=["model", alt.Tooltip("slowed:Q", format=".0f"),
                                      alt.Tooltip("period:Q", format=".1f")]).properties(height=240),
                    width="stretch")
    c2.altair_chart(a.encode(y=alt.Y("amplitude:Q", title="amplitude relative to control"),
                             tooltip=["model", alt.Tooltip("slowed:Q", format=".0f"),
                                      alt.Tooltip("amplitude:Q", format=".2f")]).properties(height=240),
                    width="stretch")
    note("At these base parameters, slowing protein removal (as a CK1 inhibitor does) lengthens the period of "
         "the two linear models and shortens the saturating one. It looks like a decisive test, but it is not: "
         "re-tuned to the switch's time course, the saturating model lengthens its period by almost exactly the "
         "switch's amounts. Which experiment? refits every rival to find what really separates them.")

# ---------------------------------------------------------------- explore
with tabs[1]:
    left, right = st.columns([1, 2], gap="large")
    with left:
        key = st.radio("Model", list(MODELS), format_func=NAMES.get, horizontal=True)
        m = MODELS[key]
        eyebrow("Parameters (multiples of the base set)")
        mults = [0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0, 2.8, 4.0]
        theta = []
        for name, value in zip(m.param_names, m.base):
            mult = st.select_slider(f"{name}: {m.param_notes.get(name, '')}", mults, value=1.0,
                                    key=f"{key}-{name}")
            theta.append(value * mult)
        eyebrow("Perturbation")
        inhibitor = st.slider("Protein removal (× normal)", 0.5, 1.0, 1.0, 0.05)
        dosage = st.slider("Transcription (× normal)", 0.25, 2.0, 1.0, 0.05)
    with right:
        base = cycle(key, tuple(theta))
        pert = cycle(key, tuple(theta), inhibitor, dosage) if (inhibitor, dosage) != (1.0, 1.0) else None
        if base is None:
            callout("<b>No oscillation.</b> With these parameters the loop settles to a steady level. Try steeper "
                    "repression (hill) or more gain.")
        else:
            cols = st.columns(3)
            cols[0].markdown(kpi("Period", f"{base['period']:.1f} h", "unperturbed"), unsafe_allow_html=True)
            cols[1].markdown(kpi("mRNA swing", f"{np.exp(base['amplitude']):,.0f}-fold", "peak to trough"),
                             unsafe_allow_html=True)
            if (inhibitor, dosage) != (1.0, 1.0):
                val = "stops" if pert is None else f"{pert['period']:.1f} h"
                txt = ("the clock stops" if pert is None else
                       f"amplitude × {pert['amplitude'] / base['amplitude']:.2f}")
                cols[2].markdown(kpi("Perturbed period", val, txt), unsafe_allow_html=True)
            st.write("")
            rows = []
            for label, c in (("control", base), ("perturbed", pert)):
                if c is None:
                    continue
                for species, sl in (("mrna", "mRNA"), ("protein", "Total protein")):
                    t, y = waveform(c, species, 72)
                    rows += [{"hours": a, "level": b, "condition": label, "species": sl} for a, b in zip(t, y)]
            df = pd.DataFrame(rows)
            legend([("control", COLOURS[key])] + ([("perturbed (dashed)", NEUTRAL)] if pert is not None else []))
            for species in ("mRNA", "Total protein"):
                eyebrow(species)
                chart = alt.Chart(df[df["species"] == species]).mark_line(strokeWidth=2).encode(
                    x=alt.X("hours:Q", axis=alt.Axis(values=list(range(0, 73, 12)))),
                    y=alt.Y("level:Q", title="log level (relative)"),
                    color=alt.Color("condition:N", scale=alt.Scale(domain=["control", "perturbed"],
                                                                   range=[COLOURS[key], NEUTRAL]),
                                    legend=None),
                    strokeDash=alt.StrokeDash("condition:N", scale=alt.Scale(domain=["control", "perturbed"],
                                                                             range=[[1, 0], [5, 3]]), legend=None),
                ).properties(height=210)
                st.altair_chart(chart, width="stretch")
        note("The model runs live: each change integrates the equations until the oscillation settles.")

# ---------------------------------------------------------------- which experiment
with tabs[2]:
    if disc is None:
        st.info("Run scripts/run_design.py to compute this.")
    else:
        st.markdown(
            "Take one model as the truth, compute exactly what it predicts for an experiment, and refit each "
            "rival to those predictions. **Δ** is how far the best version of the rival still misses, in units "
            "of measurement error squared. It is the evidence the experiment is expected to give against the "
            "rival: below 2 the two are indistinguishable, above 10 the rival is decisively rejected, and "
            "repeating the experiment doubles it.")
        d = disc.assign(pair=disc["truth"].map(NAMES) + " → " + disc["rival"].map(NAMES),
                        design_name=disc["design"].map(short),
                        shown=disc["delta"].map(lambda v: f"{v:,.0f}" if v >= 10 else f"{v:.1f}"))
        order = [short(k) for k in DESIGNS]
        heat = alt.Chart(d).encode(
            x=alt.X("design_name:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
            y=alt.Y("pair:N", title="truth → rival", axis=alt.Axis(labelLimit=320)))
        rect = heat.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("delta:Q", scale=alt.Scale(type="symlog", scheme="blues", constant=2),
                            legend=alt.Legend(title="Δ")),
            tooltip=["pair", "design_name", alt.Tooltip("delta:Q", format=",.2f")])
        text = heat.mark_text(fontSize=12).encode(
            text="shown:N",
            color=alt.condition(alt.datum.delta > 200, alt.value("white"), alt.value(INK)))
        st.altair_chart((rect + text).properties(height=260), width="stretch")
        note("Rows: which model generated the data → which rival tried to mimic it. Each rival was refitted "
             "from many starting points, so Δ is what no setting of its parameters can mimic.")

        if ident is not None:
            eyebrow("With noise: simulated experiments")
            metric = st.radio("Count a data set as a success when", ["decisive", "identified"], horizontal=True,
                              format_func={"decisive": "the data rule out both rivals",
                                           "identified": "BIC picks the right model"}.get)
            s = ident.assign(design_name=ident["design"].map(short), truth_name=ident["truth"].map(NAMES),
                             pct=100 * ident[metric])
            bars = alt.Chart(s).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("design_name:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
                xOffset=alt.XOffset("truth_name:N", sort=[NAMES[k] for k in MODELS]),
                y=alt.Y("pct:Q", title="simulated data sets (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("truth_name:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title="truth")),
                tooltip=["truth_name", "design_name", alt.Tooltip("pct:Q", format=".0f", title="%"), "reps",
                         alt.Tooltip("chosen:N", title="BIC chose")])
            st.altair_chart(bars.properties(height=260), width="stretch")
            if metric == "decisive":
                note(f"{int(ident['reps'].iloc[0])} simulated data sets per bar, each fitted by all three models. "
                     "Success: the true model fits better than both rivals by at least 10 in SSE (a likelihood "
                     "ratio of about 150), so the data themselves rule the rivals out.")
            else:
                note("BIC also penalises parameters, so when all three models fit equally it picks the one with "
                     "fewest: the delay chain (4), then the switch (5), then saturation (7). A high bar where the "
                     "other measure is low is a correct guess for the wrong reason.")

# ---------------------------------------------------------------- identifiability
with tabs[3]:
    if info is None:
        st.info("Run scripts/run_design.py to compute this.")
    else:
        key = st.radio("Model", list(MODELS), format_func=NAMES.get, horizontal=True, key="info-model")
        m = MODELS[key]
        st.markdown(
            "Even within the right model, a time course pins down only a few combinations of the parameters. "
            "The Fisher information says how precisely each combination is determined; its spread over many "
            "orders of magnitude is what is called a *sloppy* model.")
        rows = []
        for d in DESIGNS:
            w = np.array(info[f"{key}/{d}"]["eigenvalues"])
            for i, v in enumerate(w):
                rows.append({"design": short(d), "sd": 1 / np.sqrt(max(v, 1e-12)), "rank": i + 1})
        df = pd.DataFrame(rows)
        order = [short(k) for k in DESIGNS]
        ticks = alt.Chart(df).mark_tick(thickness=3, size=36, color=COLOURS[key]).encode(
            x=alt.X("design:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
            y=alt.Y("sd:Q", scale=alt.Scale(type="log", domain=[1e-3, 1e4]),
                    axis=alt.Axis(values=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], format="~g"),
                    title="sd of each parameter combination (0.1 = 10%)"),
            tooltip=["design", alt.Tooltip("sd:Q", format=".3g"), "rank"])
        rule = alt.Chart(pd.DataFrame({"y": [0.1]})).mark_rule(strokeDash=[5, 3], color=INK).encode(y="y:Q")
        st.altair_chart((ticks + rule).properties(height=320), width="stretch")
        note("Each tick is one combination of the model's log parameters (an eigenvector of the Fisher "
             "information). Below the dashed line it is determined to within 10%.")
        eyebrow("Precision of each parameter on its own (± %, 1 sd)")
        tab = {}
        for d in DESIGNS:
            se = np.array(info[f"{key}/{d}"]["se_log"])
            tab[short(d)] = [("undetermined" if not np.isfinite(s) or s > 3 else f"± {100 * (np.exp(s) - 1):,.0f}%")
                             for s in se]
        st.dataframe(pd.DataFrame(tab, index=[f"{n}: {m.param_notes.get(n, '')}" for n in m.param_names]),
                     width="stretch")

# ---------------------------------------------------------------- real data
with tabs[4]:
    if real is None:
        st.info("Run scripts/fit_real_data.py (with the GEO data downloaded) to compute this.")
    else:
        fits = pd.read_csv(RESULTS / "real_fit.csv")
        t0 = real["ct_start"]
        st.markdown(
            f"Hughes et al. (2009) measured gene expression in the livers of mice kept in constant darkness, "
            f"every hour for two days (GEO GSE11923). Here are {real['gene']} mRNA ({real['probe']}) and each "
            f"model's best fit; measurement noise is estimated at ± {100 * (np.exp(real['sigma']) - 1):.0f}%.")
        pts = pd.DataFrame({"ct": np.array(real["data"]["t"]) + t0, "level": real["data"]["y"]})
        lines = fits.melt(id_vars="t", var_name="key", value_name="level")
        lines = lines.assign(ct=lines["t"] + t0, model=lines["key"].map(NAMES))
        chart = (alt.Chart(lines).mark_line(strokeWidth=2).encode(
            x=alt.X("ct:Q", title="circadian time (hours)", scale=alt.Scale(domain=[t0, t0 + 47], nice=False),
                    axis=alt.Axis(values=list(range(t0, t0 + 48, 6)))),
            y=alt.Y("level:Q", title="log expression", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title=None)))
            + alt.Chart(pts).mark_circle(size=34, color=INK, opacity=0.9).encode(
                x="ct:Q", y="level:Q", tooltip=[alt.Tooltip("ct:Q"), alt.Tooltip("level:Q", format=".2f")]))
        st.altair_chart(chart.properties(height=280), width="stretch")
        tab = pd.DataFrame({NAMES[k]: {"Misfit (SSE)": f"{v['sse']:.1f}", "BIC": f"{v['bic']:.1f}",
                                       "Posterior probability": f"{v['posterior']:.2f}",
                                       "Period": f"{v['period']:.1f} h"} for k, v in real["models"].items()})
        st.dataframe(tab, width="stretch")

        eyebrow("What each fitted model predicts for a CK1 inhibitor")
        pr = pd.read_csv(RESULTS / "real_predictions.csv")
        pr = pr.assign(change=pr["period"] - pr["control_period"], model=pr["model"].map(NAMES))
        labels = [Perturbation(inhibitor=f).label for f in INHIBITOR_DOSES]
        rows = []
        for (mname, lab), g in pr[pr["perturbation"].isin(labels)].groupby(["model", "perturbation"]):
            ch = g["change"].dropna()
            if len(ch):
                f = INHIBITOR_DOSES[labels.index(lab)]
                rows.append({"model": mname, "dose": f"slowed {100 * (1 - f):.0f}%", "lo": ch.quantile(0.05),
                             "mid": ch.median(), "hi": ch.quantile(0.95)})
        df = pd.DataFrame(rows)
        dose_order = [f"slowed {100 * (1 - f):.0f}%" for f in INHIBITOR_DOSES]
        base = alt.Chart(df).encode(
            x=alt.X("dose:N", sort=dose_order, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("model:N", sort=[NAMES[k] for k in MODELS]),
            color=alt.Color("model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title=None)))
        bars = base.mark_bar(size=18, opacity=0.85).encode(
            y=alt.Y("lo:Q", title="change in period (hours)"), y2="hi:Q",
            tooltip=["model", "dose", alt.Tooltip("mid:Q", format="+.1f", title="median"),
                     alt.Tooltip("lo:Q", format="+.1f", title="5%"), alt.Tooltip("hi:Q", format="+.1f", title="95%")])
        mid = base.mark_tick(color="white", thickness=2, size=18).encode(y="mid:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK).encode(y="y:Q")
        st.altair_chart((bars + mid + zero).properties(height=280), width="stretch")
        note("90% intervals over every parameter set consistent with the liver data (posterior samples), not "
             "just the best fit.")

# ---------------------------------------------------------------- methodology
with tabs[5]:
    if METHODOLOGY_FILE.exists():
        st.markdown(METHODOLOGY_FILE.read_text().replace(
            "](../results/", "](https://raw.githubusercontent.com/jackcormacwright1/telling-clocks-apart/main/results/"))
