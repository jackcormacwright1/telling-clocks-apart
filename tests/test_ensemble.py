import numpy as np

from clocks.ensemble import effective_size, predict_rhythms, sample
from clocks.experiments import DESIGNS, PERTURBATIONS, simulate
from clocks.fit import fit
from clocks.models import DELAY


def test_effective_size_of_independent_and_correlated_draws():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=2000)
    assert 1500 < effective_size(iid) < 2500
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    assert effective_size(ar) < 300   # theory: 2000 x 0.1 / 1.9, about 105


def test_sampler_stays_in_bounds_and_near_the_fit():
    data = simulate(DELAY, DELAY.base, DESIGNS["standard"], rng=np.random.default_rng(2))
    f = fit(DELAY, data, n_random=0)
    ens = sample(DELAY, data, f, n_steps=900, burn=600, thin=10, rng=np.random.default_rng(1))
    lo, hi = DELAY.log_bounds()
    z = ens.samples[:, :DELAY.n_params]
    assert len(ens.samples) == 30
    assert np.all(z >= lo) and np.all(z <= hi)
    assert 0.02 < ens.acceptance < 0.9
    # draws fit about as well as the best fit, give or take a few parameters' worth of chi-squared
    assert np.median(ens.sse) < f.sse + 15
    P = predict_rhythms(ens, PERTURBATIONS[:1])
    assert np.nanmedian(P[:, 0]) == np.nanmedian(P[:, 0])  # some draws oscillate
    assert np.all(P[~np.isnan(P[:, 1]), 1] > P[~np.isnan(P[:, 1]), 0])  # the inhibitor lengthens the delay clock
