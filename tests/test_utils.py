import numpy as np
import pytest
from scipy.stats import norm

from smbh_corona.constants import arcsec, c, k_B, mJy, pi
from smbh_corona.utils import (
    GaussianLikelihood_withUL_andLL,
    Omega_beam,
    Sobs_to_TB,
    get_f_sys,
)


def test_brightness_temperature_and_beam_area_use_cgs_units():
    flux_mjy = np.array([1.0])
    frequency_hz = np.array([1e9])
    beam_sr = np.array([1e-12])
    expected_temperature = c**2 * mJy / (2 * k_B * frequency_hz**2 * beam_sr)

    assert Sobs_to_TB(flux_mjy, frequency_hz, beam_sr)[0] == pytest.approx(
        expected_temperature[0]
    )
    assert Omega_beam(1e-3, 1e-3) == pytest.approx(
        pi * 1e-6 / (4 * np.log(2))
    )


def test_systematic_error_fractions_for_representative_bands():
    assert get_f_sys(100, 1 * arcsec, 1 * arcsec) == 0.05
    assert get_f_sys(300, 0.4 * arcsec, 0.4 * arcsec) == 0.10
    assert get_f_sys(700, 0.4 * arcsec, 0.4 * arcsec) == 0.20


def test_soft_mixed_limits_use_explicit_parameters_and_w(monkeypatch):
    from bilby.core import likelihood as bilby_likelihood

    monkeypatch.setattr(
        bilby_likelihood, "PARAMETERS_AS_STATE", "FALSE", raising=False
    )

    def model(x, offset):
        return np.asarray(x) * 0 + offset

    likelihoods = [
        GaussianLikelihood_withUL_andLL(
            x=[0.0],
            y=[0.1],
            func=model,
            sigma=0.3,
            x_ul=[0.0],
            y_ul=[0.5],
            sigma_ul=0.2,
            x_ll=[0.0],
            y_ll=[-0.5],
            sigma_ll=0.2,
            w=w,
        )
        for w in (1.0, 10.0)
    ]
    parameters = {"offset": 0.8}
    log_likelihoods = [likelihood.log_likelihood(parameters) for likelihood in likelihoods]

    expected_detection = -(0.1 - 0.8) ** 2 / (2 * 0.3**2) - np.log(
        2 * np.pi * 0.3**2
    ) / 2
    expected_soft = (
        expected_detection
        + norm.logcdf((0.5 - 0.8) / 0.2)
        + norm.logcdf((0.8 + 0.5) / 0.2)
    )
    assert log_likelihoods[0] == pytest.approx(expected_soft)
    assert log_likelihoods[1] < log_likelihoods[0]
