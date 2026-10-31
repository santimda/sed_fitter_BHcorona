"""Empirical millimeter, X-ray, and bolometric luminosity relations.

All luminosities in this module are luminosities in ``erg/s``.  The
millimeter quantities are ``nu * L_nu`` at the frequency in the function
name, rather than monochromatic ``L_nu``.
"""

import warnings
import numpy as np

from smbh_corona.constants import mJy, pc, pi


R23_SCATTER_DEX = 0.22
K22_SCATTER_DEX = 0.40
V26A_XRAY_SCATTER_DEX = 0.28
V26A_BOLOMETRIC_SCATTER_DEX = 0.35


def flux_to_nu_luminosity(flux_mjy, distance_mpc, frequency_ghz, magnification=1.0):
    """Convert flux density to intrinsic ``nu * L_nu`` in ``erg/s``.

    ``flux_mjy`` is in mJy, ``distance_mpc`` in Mpc, and ``frequency_ghz``
    in GHz.  A magnification greater than one is removed from the result.
    """
    flux_mjy = _positive_array(flux_mjy, "flux_mjy")
    distance_mpc = _positive_array(distance_mpc, "distance_mpc")
    frequency_ghz = _positive_array(frequency_ghz, "frequency_ghz")
    magnification = _positive_array(magnification, "magnification")
    return (
        4
        * pi
        * (distance_mpc * 1e6 * pc) ** 2
        * (frequency_ghz * 1e9)
        * flux_mjy
        * mJy
        / magnification
    )


def _positive_array(value, name):
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _linear_relation(luminosity, slope, intercept):
    log_luminosity = np.log10(_positive_array(luminosity, "luminosity"))
    return 10 ** ((log_luminosity - intercept) / slope)


def _warn_outside_calibration(luminosity, relation, lower, upper):
    luminosity = _positive_array(luminosity, "luminosity")
    log_luminosity = np.log10(luminosity)
    if np.any((log_luminosity < lower) | (log_luminosity > upper)):
        warnings.warn(
            f"{relation} is calibrated for log10 luminosity in "
            f"[{lower:g}, {upper:g}]; extrapolating outside this range",
            RuntimeWarning,
            stacklevel=3,
        )


def R23_L100_to_LX2(L100):
    """Return R23 ``L_2-10 keV`` from rest-frame, de-magnified ``L_100 GHz``."""
    _warn_outside_calibration(L100, "R23", 36, 41)
    return _linear_relation(L100, slope=1.22, intercept=-13.9)


def R23_L100_to_LX(L100):
    """Return R23 ``L_14-150 keV`` from rest-frame, de-magnified ``L_100 GHz``."""
    _warn_outside_calibration(L100, "R23", 36, 41)
    return _linear_relation(L100, slope=1.22, intercept=-14.4)


def K22_L230_to_LX2(L230):
    """Return K22 ``L_2-10 keV`` from rest-frame, de-magnified ``L_230 GHz``."""
    _warn_outside_calibration(L230, "K22", 37, 41)
    return _linear_relation(L230, slope=1.08, intercept=-7.41)


def K22_L230_to_LX(L230):
    """Return K22 ``L_14-150 keV`` from rest-frame, de-magnified ``L_230 GHz``."""
    _warn_outside_calibration(L230, "K22", 37, 41)
    return _linear_relation(L230, slope=1.19, intercept=-12.78)


def R23_F100_to_FX(F100):
    """Return R23 ``F_14-150 keV`` from ``nu*F_nu`` at 100 GHz."""
    return _linear_relation(F100, slope=1.37, intercept=-1.3)


def K22_F230_to_FX(F230):
    """Return K22 ``F_14-150 keV`` from ``nu*F_nu`` at 230 GHz."""
    return _linear_relation(F230, slope=1.17, intercept=-2.73)


def V26a_L100_to_LX2(L100):
    """Return ``L_2-10 keV`` from ``L_100 GHz`` using Venselaar et al. 2026a.

    The relation is expressed as ``log10(L100) - 38 =
    0.15*x**2 + 1.20*x + 0.54``, where ``x = log10(L2-10) - 43``.
    """
    _warn_outside_calibration(L100, "V26a", 36, 41)
    log_l100 = np.log10(_positive_array(L100, "L100"))
    y = log_l100 - 38.0
    a, b, c = 0.15, 1.20, 0.54
    discriminant = b**2 - 4 * a * (c - y)
    if np.any(discriminant < 0):
        raise ValueError("L100 is outside the real-valued V26a relation domain")
    x = (-b + np.sqrt(discriminant)) / (2 * a)
    return 10 ** (x + 43.0)


def V26a_LX2_to_L100(LX2):
    """Invert the single V26a relation for plotting and round-trip checks."""
    log_lx2 = np.log10(_positive_array(LX2, "LX2"))
    x = log_lx2 - 43.0
    return 10 ** (0.15 * x**2 + 1.20 * x + 0.54 + 38.0)


def L100_to_Lbol(L100):
    """Return bolometric luminosity from rest-frame, de-magnified ``L_100 GHz``."""
    _warn_outside_calibration(L100, "V26a", 36, 41)
    log_l100 = np.log10(_positive_array(L100, "L100"))
    return 10 ** ((log_l100 + 6.22) / 1.01)
