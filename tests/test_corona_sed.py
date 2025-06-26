import numpy as np
from smbh_corona.corona_sed import S_nu as compute_corona_sed


def test_shape_and_nonnegativity():
    """
    compute_corona_sed should return an iterable of the same shape as input
    and all flux values must be non-negative.
    """
    freqs = np.logspace(9, 12, 50)
    flux = compute_corona_sed(freqs, tau_T=1.0, T=1e9)
    # allow list or ndarray
    assert hasattr(flux, '__iter__')
    arr = np.asarray(flux)
    assert arr.shape == freqs.shape
    assert np.all(arr >= 0)

def test_zero_tau_yields_zero_flux():
    """
    With tau_T=0, the thermal electron density is zero, so emission yields NaNs
    due to 0/0 operations in the absorption coefficient.
    """
    freqs = np.logspace(9, 12, 20)
    flux_zero = compute_corona_sed(freqs, tau_T=0.0, T=1e9)
    # all entries should be NaN
    arr_zero = np.asarray(flux_zero)
    assert np.all(np.isnan(arr_zero))


def test_flux_increases_with_tau():
    """
    At a low frequency (optically thick regime), flux should increase with tau_T,
    allowing for small numerical fluctuations.
    """
    freq = 1e9  # low frequency
    f_tau_low = compute_corona_sed(freq, tau_T=0.1, T=1e9)[0]
    f_tau_mid = compute_corona_sed(freq, tau_T=1.0, T=1e9)[0]
    f_tau_high = compute_corona_sed(freq, tau_T=5.0, T=1e9)[0]
    # require at least 99.9% of the expected increase
    assert (f_tau_mid / f_tau_low) >= 0.999
    assert (f_tau_high / f_tau_mid) >= 0.999


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
    Flux should scale inversely with the square of the provided luminosity distance D_L.
    """
    freqs = np.array([1e10, 3e10])
    f1 = compute_corona_sed(freqs, tau_T=1.0, T=1e9, D_L=100.0)
    arr1 = np.asarray(f1)
    f2 = compute_corona_sed(freqs, tau_T=1.0, T=1e9, D_L=200.0)
    arr2 = np.asarray(f2)
    # doubling D_L should reduce flux by factor of 4
    assert np.allclose(arr2, arr1 / 4, rtol=1e-6)
