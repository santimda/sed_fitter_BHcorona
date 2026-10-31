import numpy as np
import pytest

from smbh_corona.correlations import (
    K22_F230_to_FX,
    K22_L230_to_LX,
    K22_L230_to_LX2,
    L100_to_Lbol,
    R23_F100_to_FX,
    R23_L100_to_LX,
    R23_L100_to_LX2,
    V26a_L100_to_LX2,
    V26a_LX2_to_L100,
    flux_to_nu_luminosity,
)


def test_empirical_relations_match_published_coefficients():
    assert np.log10(R23_L100_to_LX2(1e40)) == pytest.approx((40 + 13.9) / 1.22)
    assert np.log10(R23_L100_to_LX(1e40)) == pytest.approx((40 + 14.4) / 1.22)
    assert np.log10(K22_L230_to_LX2(1e40)) == pytest.approx((40 + 7.41) / 1.08)
    assert np.log10(K22_L230_to_LX(1e40)) == pytest.approx((40 + 12.78) / 1.19)
    assert np.log10(R23_F100_to_FX(1e-12)) == pytest.approx((-12 + 1.3) / 1.37)
    assert np.log10(K22_F230_to_FX(1e-12)) == pytest.approx((-12 + 2.73) / 1.17)
    assert np.log10(L100_to_Lbol(1e40)) == pytest.approx((40 + 6.22) / 1.01)


def test_v26a_relation_round_trips_for_scalar_and_array_inputs():
    l100 = np.array([1e38, 1e39, 1e40, 1e41])
    lx2 = V26a_L100_to_LX2(l100)

    assert np.allclose(V26a_LX2_to_L100(lx2), l100)


def test_correlations_warn_outside_range_and_reject_nonpositive_input():
    with pytest.warns(RuntimeWarning, match="R23 is calibrated"):
        R23_L100_to_LX2(1e42)
    with pytest.raises(ValueError):
        R23_L100_to_LX2(0)


def test_flux_to_nu_luminosity_uses_cgs_units_and_removes_magnification():
    luminosity = flux_to_nu_luminosity(1.0, 1.0, 100.0)
    expected = 4 * np.pi * (1e6 * 3.0857e18) ** 2 * 100e9 * 1e-26

    assert luminosity == pytest.approx(expected)
    assert flux_to_nu_luminosity(
        1.0, 1.0, 100.0, magnification=2.0
    ) == pytest.approx(expected / 2)
