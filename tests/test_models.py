import numpy as np
import pytest

from clocks.models import DELAY, MODELS, NONE, SATURATION, SWITCH, Perturbation, limit_cycle, rescale_time


@pytest.mark.parametrize("key", list(MODELS))
def test_base_parameters_give_a_24_hour_clock_with_matched_amplitude(key):
    m = MODELS[key]
    lc = limit_cycle(m, m.base)
    assert lc is not None
    assert lc.period == pytest.approx(24.0, abs=0.02)
    assert lc.amplitude == pytest.approx(3.84, abs=0.02)
    lo, hi = m.log_bounds()
    assert np.all(np.log(m.base) >= lo) and np.all(np.log(m.base) <= hi)


@pytest.mark.parametrize("key", list(MODELS))
def test_rescaling_time_divides_the_period_and_keeps_the_waveform(key):
    m = MODELS[key]
    lc = limit_cycle(m, m.base)
    fast = limit_cycle(m, rescale_time(m, m.base, 1.25))
    assert fast.period == pytest.approx(lc.period / 1.25, rel=2e-3)
    assert fast.amplitude == pytest.approx(lc.amplitude, abs=1e-2)
    assert np.max(np.abs((fast.log_mrna - fast.log_mrna.mean()) - (lc.log_mrna - lc.log_mrna.mean()))) < 0.02


def test_gentle_repression_without_saturation_or_delay_does_not_oscillate():
    theta = np.array(SWITCH.base)
    theta[SWITCH.param_names.index("hill")] = 6.0   # Griffith: three equal-ish stages need n > 8
    assert limit_cycle(SWITCH, theta) is None


def test_the_delay_chain_oscillates_with_gentle_repression_but_not_with_too_little_gain():
    theta = np.array(DELAY.base)
    theta[DELAY.param_names.index("gain")] = 3.0    # below the secant-condition threshold for n = 3
    assert limit_cycle(DELAY, theta) is None


def test_slowing_protein_removal_lengthens_linear_clocks_and_shortens_the_saturating_one():
    slow = Perturbation(inhibitor=0.8)
    for m in (SWITCH, DELAY):
        assert limit_cycle(m, m.base, slow).period > 26.0
    assert limit_cycle(SATURATION, SATURATION.base, slow).period < 22.0


def test_limit_cycle_interpolation_is_periodic():
    lc = limit_cycle(DELAY, DELAY.base)
    t = np.linspace(0, 24, 7)
    assert np.allclose(lc.at("mrna", t, 3.0), lc.at("mrna", t + lc.period, 3.0))
    assert lc.at("mrna", np.array([0.0]), 0.0)[0] == pytest.approx(lc.log_mrna[0])


def test_perturbation_labels():
    assert NONE.label == "unperturbed"
    assert Perturbation(inhibitor=0.7, dosage=0.5).label == "protein removal x0.7, transcription x0.5"
