import numpy as np
from smbh_corona.corona_sed import S_nu as compute_corona_sed
import smbh_corona.thermalsyn as thermalsyn


def test_shape_and_nonnegativity():
    """
    compute_corona_sed should return an iterable of the same shape as input
    and all flux values must be non-negative.
    """
    freqs = np.logspace(9, 12, 50)
    flux = compute_corona_sed(freqs, tau_T=1.0, T=1e9)
    # allow list or ndarray
    assert hasattr(flux, "__iter__")
    arr = np.asarray(flux)
    assert arr.shape == freqs.shape
    assert np.all(arr >= 0)


def test_flux_increases_with_r_c():
    """
    Flux should increase r_c**2 at low frequencies,
    where the emission is optically thick.
    """
    freq = 1e9  # low frequency
    r_c_low, r_c_high = 20, 50
    f_r_c_low = compute_corona_sed(freq, r_c=r_c_low)[0]
    f_r_c_high = compute_corona_sed(freq, r_c=r_c_high)[0]

    expected_ratio = (r_c_high / r_c_low) ** 2
    actual_ratio = f_r_c_high / f_r_c_low
    assert np.isclose(expected_ratio, actual_ratio, rtol=1e-3)


def test_scalar_input_handles_properly():
    """
    Passing a single frequency should return a length-1 array.
    """
    freq = 1e11
    flux = compute_corona_sed(freq, tau_T=1.0, T=1e9)
    arr = np.asarray(flux)
    assert arr.shape == (1,)


def test_distance_scaling():
    """
    Flux should scale inversely with the square of the luminosity distance D_L.
    """
    freqs = np.array([1e10, 3e10])
    f1 = compute_corona_sed(freqs, D_L=100.0)
    arr1 = np.asarray(f1)
    f2 = compute_corona_sed(freqs, D_L=200.0)
    arr2 = np.asarray(f2)
    # doubling D_L should reduce flux by factor of 4
    assert np.allclose(arr2, arr1 / 4, rtol=1e-6)


def test_thermal_scalar_memoization_caches_repeated_values():
    thermalsyn.a_fun.cache_clear()

    value = thermalsyn.a_fun(0.5)
    assert thermalsyn.a_fun(0.5) == value
    assert thermalsyn.a_fun.cache_info().hits == 1
    assert thermalsyn.a_fun.cache_info().misses == 1


def test_lnu_fun_reuses_thermal_frequency_integral(monkeypatch):
    original_i_of_x = thermalsyn.I_of_x
    calls = []

    def counted_i_of_x(x):
        calls.append(np.asarray(x).copy())
        return original_i_of_x(x)

    monkeypatch.setattr(thermalsyn, "I_of_x", counted_i_of_x)
    frequencies = np.geomspace(1e10, 1e12, 8)
    luminosity = thermalsyn.Lnu_fun(
        frequencies,
        ne=1e10,
        R=1e14,
        Theta=0.1,
        B=10.0,
        t=1e5,
    )

    assert len(calls) == 1
    assert np.asarray(luminosity).shape == frequencies.shape
    assert np.all(np.isfinite(luminosity))
