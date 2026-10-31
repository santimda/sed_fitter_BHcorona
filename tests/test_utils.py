# tests/test_utils.py

import numpy as np
import pytest
from scipy.stats import norm

from smbh_corona.utils import (
    Sobs_to_TB,
    Omega_beam,
    reduced_chi_square,
    get_f_sys,
    calculate_logX,
    L230_to_LX2,
    L230_to_LX,
    L100_to_LX2,
    L100_to_LX,
    LX_to_Lbol,
    GaussianLikelihood_withUL_andLL,
)
from smbh_corona.constants import c, mJy, k_B, pi, arcsec


def test_Sobs_to_TB_known_value():
    # TB = c^2 * (S_nu*mJy) / (2*k_B*nu^2*Omega_b)
    S_nu = np.array([1.0])  # mJy
    nu = np.array([1e9])  # Hz
    Omega_b = np.array([1e-12])  # sr
    expected = c**2 * (1.0 * mJy) / (2 * k_B * (1e9) ** 2 * 1e-12)
    tb = Sobs_to_TB(S_nu, nu, Omega_b)
    assert tb.shape == (1,)
    assert tb[0] == pytest.approx(expected)


def test_Omega_beam_symmetry_and_value():
    theta = 1e-3  # rad
    omega = Omega_beam(theta, theta)
    manual = pi * theta * theta / (4 * np.log(2))
    assert omega == pytest.approx(manual)


def test_reduced_chi_square_perfect_fit():
    obs = np.array([1.0, 2.0, 3.0])
    model = obs.copy()
    errs = np.ones_like(obs)
    # chi2 = 0, dof = 3 - 0 = 3
    rc = reduced_chi_square(obs, model, errs, num_params=0)
    assert rc == pytest.approx(0.0)


def test_get_f_sys_default_and_alma_bands():
    # Default (low resolution)
    assert get_f_sys(100, 1 * arcsec, 1 * arcsec) == 0.05
    # High res + band 212-500
    assert get_f_sys(300, 0.4 * arcsec, 0.4 * arcsec) == 0.10
    # High res + band 600-900
    assert get_f_sys(700, 0.4 * arcsec, 0.4 * arcsec) == 0.20


def test_calculate_logX_and_inverse_relations():
    alpha, beta = 2.0, -1.0
    x = calculate_logX(1.0, alpha, beta)
    # Should satisfy alpha*log10(x) + beta = 1 -> log10(x) = (1-beta)/alpha
    assert x == pytest.approx((1.0 - beta) / alpha)


def test_x_to_x_relations_consistency():
    # Use a sample luminosity
    L = 1e44
    logL = np.log10(L)
    # Test L230->LX2
    lx2 = L230_to_LX2(L)
    alpha, beta = 1.08, -7.41
    assert np.log10(lx2) == pytest.approx((logL - beta) / alpha)
    # Other conversions should yield positive values
    assert L230_to_LX(L) > 0
    assert L100_to_LX2(L) > 0
    assert L100_to_LX(L) > 0
    assert LX_to_Lbol(1e43) == pytest.approx(8.48e43)


def test_vectorization_and_shapes():
    arr = np.array([1.0, 2.0, 3.0])
    # All functions should broadcast over arrays
    assert Sobs_to_TB(arr, arr * 1e9, arr * 1e-12).shape == arr.shape
    assert np.all(Omega_beam(arr, arr) > 0)
    assert np.allclose(calculate_logX(np.log10(arr), 1.0, 0.0), np.log10(arr))


def test_smooth_limits_are_softened_by_w(monkeypatch):
    from bilby.core import likelihood as bilby_likelihood

    monkeypatch.setattr(bilby_likelihood, "PARAMETERS_AS_STATE", "FALSE")

    def model(x, offset=0.0):
        return np.asarray(x) * 0 + offset

    likelihood_soft = GaussianLikelihood_withUL_andLL(
        [0.0], [0.0], model, sigma=1.0,
        x_ul=[0.0], y_ul=[1.0], sigma_ul=0.1, w=1.0,
        x_ll=[0.0], y_ll=[-1.0], sigma_ll=0.1,
    )
    likelihood_strict = GaussianLikelihood_withUL_andLL(
        [0.0], [0.0], model, sigma=1.0,
        x_ul=[0.0], y_ul=[1.0], sigma_ul=0.1, w=10.0,
        x_ll=[0.0], y_ll=[-1.0], sigma_ll=0.1,
    )
    parameters = {"offset": 1.2}
    assert likelihood_strict.log_likelihood(parameters) < likelihood_soft.log_likelihood(
        parameters
    )


def test_likelihood_uses_sampled_sigma_from_parameters(monkeypatch):
    from bilby.core import likelihood as bilby_likelihood

    monkeypatch.setattr(bilby_likelihood, "PARAMETERS_AS_STATE", "FALSE")

    def model(x, offset):
        return np.asarray(x) * 0 + offset

    likelihood = GaussianLikelihood_withUL_andLL([0.0], [1.0], model, sigma=None)
    log_likelihood = likelihood.log_likelihood({"offset": 0.0, "sigma": 2.0})

    expected = -(1.0 / 2.0) ** 2 / 2 - np.log(2 * np.pi * 2.0**2) / 2
    assert log_likelihood == pytest.approx(expected)


def test_likelihood_batches_detections_and_limits_into_one_model_call(monkeypatch):
    from bilby.core import likelihood as bilby_likelihood

    monkeypatch.setattr(bilby_likelihood, "PARAMETERS_AS_STATE", "FALSE")
    calls = []

    def model(x, offset):
        x = np.asarray(x)
        calls.append(x.copy())
        return 2 * x + offset

    likelihood = GaussianLikelihood_withUL_andLL(
        x=[1.0, 2.0],
        y=[2.0, 3.0],
        func=model,
        sigma=0.5,
        x_ul=[3.0],
        y_ul=[6.5],
        sigma_ul=0.4,
        x_ll=[4.0],
        y_ll=[6.0],
        sigma_ll=0.3,
        w=2.0,
    )

    actual = likelihood.log_likelihood({"offset": 0.25})
    detection_predictions = 2 * np.array([1.0, 2.0]) + 0.25
    upper_prediction = 2 * 3.0 + 0.25
    lower_prediction = 2 * 4.0 + 0.25
    expected = np.sum(
        -((np.array([2.0, 3.0]) - detection_predictions) / 0.5) ** 2 / 2
        - np.log(2 * np.pi * 0.5**2) / 2
    )
    expected += norm.logcdf((6.5 - upper_prediction) / (0.4 / 2.0))
    expected += norm.logcdf((lower_prediction - 6.0) / (0.3 / 2.0))

    assert actual == pytest.approx(expected)
    assert len(calls) == 1
    assert np.array_equal(calls[0], [1.0, 2.0, 3.0, 4.0])
