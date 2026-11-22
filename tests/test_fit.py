import numpy as np
import pytest

from clocks.design import information
from clocks.experiments import DESIGNS, simulate
from clocks.fit import Problem, estimate_period, fit
from clocks.models import DELAY, SWITCH, rescale_time


def test_design_layout_matches_its_observations():
    for d in DESIGNS.values():
        data = simulate(DELAY, DELAY.base, d, phase=3.0)
        assert data.vector().size == d.n_obs == d.sigmas.size == len(d.labels())


def test_profiled_offsets_make_the_fit_blind_to_the_measurement_scale():
    d = DESIGNS["protein"]
    data = simulate(DELAY, DELAY.base, d, phase=3.0, rng=np.random.default_rng(1))
    x = np.r_[np.log(DELAY.base), 0.1]
    r1 = Problem(DELAY, data)(x)
    data.levels = [lv + 5.0 for lv in data.levels]
    assert np.allclose(Problem(DELAY, data)(x), r1)


def test_a_model_fits_its_own_noise_free_data_exactly():
    data = simulate(DELAY, DELAY.base, DESIGNS["inhibitor"], phase=5.0)
    start = rescale_time(DELAY, np.array(DELAY.base) * [1.3, 0.8, 1.1, 0.7], 1.0)
    f = fit(DELAY, data, starts=[start], n_random=0, include_base=False)
    assert f.sse < 1e-3
    assert f.period == pytest.approx(24.0, abs=0.01)
    assert f.phase == pytest.approx(5.0, abs=0.05)


def test_noise_gives_a_chi_squared_sized_misfit():
    rng = np.random.default_rng(3)
    d = DESIGNS["standard"]
    sse = [fit(DELAY, simulate(DELAY, DELAY.base, d, rng=rng), n_random=0).sse for _ in range(4)]
    # 13 observations, 4 parameters + phase + offset: about 7 degrees of freedom
    assert 1.0 < np.mean(sse) < 16.0


def test_period_estimate_from_a_time_course():
    data = simulate(SWITCH, SWITCH.base, DESIGNS["hourly"])
    assert estimate_period(data) == pytest.approx(24.0, abs=0.5)


def test_fisher_information_is_sloppy_and_grows_with_data():
    a = information("delay", "standard")
    b = information("delay", "everything")
    assert np.all(np.diff(a.eigenvalues) <= 0)
    assert a.eigenvalues[0] / a.eigenvalues[-1] > 100
    assert np.all(b.se_log <= a.se_log + 1e-9)
