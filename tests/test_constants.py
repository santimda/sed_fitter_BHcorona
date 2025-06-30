import pytest
from smbh_corona import constants

"""Check that the constants are loaded correctly"""


def test_speed_of_light_positive():
    assert constants.c > 0


def test_electron_mass_cgs_value():
    # m_e should be ~9.11e-28 g
    assert pytest.approx(constants.me / 9.11e-28, rel=1e-2) == 1.0


def test_sigma_T_value():
    # Thomson cross-section ~6.65e-25 cm²
    assert pytest.approx(constants.sigma_T / 6.65e-25, rel=1e-3) == 1.0
