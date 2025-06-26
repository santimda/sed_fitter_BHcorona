import pytest
from smbh_corona import constants

def test_speed_of_light_positive():
    assert constants.c > 0

def test_electron_mass_units():
    # m_e should be ~9.11e-31 kg
    assert pytest.approx(constants.me, rel=1e-3) == 9.109e-31

def test_sigma_T_value():
    # Thomson cross-section ~6.65e-29 m²
    assert pytest.approx(constants.sigma_T, rel=1e-3) == 6.652e-29
