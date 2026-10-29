import numpy as np
import pytest

from smbh_corona.correlations import (
    K22_L230_to_LX,
    K22_L230_to_LX2,
    L100_to_Lbol,
    R23_L100_to_LX,
    R23_L100_to_LX2,
    V26a_L100_to_LX2,
    V26a_LX2_to_L100,
    flux_to_nu_luminosity,
)


def test_linear_relations_match_published_coefficients():
    L100 = 1e40
    assert np.log10(R23_L100_to_LX2(L100)) == pytest.approx((40 + 13.9) / 1.22)
    assert np.log10(R23_L100_to_LX(L100)) == pytest.approx((40 + 14.4) / 1.22)
    L230 = 1e40
    assert np.log10(K22_L230_to_LX2(L230)) == pytest.approx((40 + 7.41) / 1.08)
    assert np.log10(K22_L230_to_LX(L230)) == pytest.approx((40 + 12.78) / 1.19)


def test_v26a_round_trip():
    L100 = np.logspace(37, 42, 20)
    LX2 = V26a_L100_to_LX2(L100)
    assert np.allclose(V26a_LX2_to_L100(LX2), L100)


def test_v26a_and_bolometric_relations_match_equations():
    L100 = 1e40
    x = np.log10(V26a_L100_to_LX2(L100)) - 43
    expected_l100 = 10 ** (0.15 * x**2 + 1.20 * x + 0.54 + 38)
    assert V26a_LX2_to_L100(V26a_L100_to_LX2(L100)) == pytest.approx(expected_l100)
    assert np.log10(L100_to_Lbol(L100)) == pytest.approx((40 + 6.22) / 1.01)


def test_correlations_vectorize_and_reject_nonpositive_values():
    values = np.array([1e38, 1e40, 1e42])
    assert R23_L100_to_LX2(values).shape == values.shape
    with pytest.raises(ValueError):
        R23_L100_to_LX2(0)


def test_flux_to_nu_luminosity_uses_cgs_units_and_demagnifies():
    luminosity = flux_to_nu_luminosity(1.0, 1.0, 100.0)
    expected = 4 * np.pi * (1e6 * 3.0857e18) ** 2 * 100e9 * 1e-26
    assert luminosity == pytest.approx(expected)
    assert flux_to_nu_luminosity(1.0, 1.0, 100.0, magnification=2.0) == pytest.approx(expected / 2)
