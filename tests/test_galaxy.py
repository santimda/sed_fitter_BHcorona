import os
import json
import numpy as np
import pytest

from smbh_corona.galaxy import Galaxy


# A dummy subclass that skips any file I/O by overriding load_parameters & load_flux_data
class DummyGalaxy(Galaxy):
    def __init__(self):
        # We don't call super().__init__, so skip file loading
        # Just set seed_parameters and update attributes
        self.use_kT_par = False
        self.use_E_cond = False
        # minimal seed params
        self.seed_parameters = {
            "z": 0.05,
            "log_M": 8.0,
            "r_c": 100.0,
            "log_delta": -1.0,
            "tau_T": 0.5,
            "p": 2.5,
            "magnification": 1.0,
            "alpha_sy": -0.7,
            "beta_RJ": 1.5,
            "tau1_freq": 200.0,
            "sy_scale": 1.0,
            "ff_scale": 1.0,
            "RJ_scale": 1.0,
            "nu_tau1_cl": 10.0,
            "f_cov": 0.2,
            "nu_tau1_diff": 1.0,
            # we'll set kT & log_eps_B directly
            "kT": 50.0,
            "log_eps_B": -1.0,
        }
        self.update_parameters(self.seed_parameters)
        # dummy distance
        from astropy.cosmology import Planck18

        self.D_L = Planck18.luminosity_distance(self.z)

    def load_parameters(self):
        raise RuntimeError("should not be called")

    def load_flux_data(self):
        # return empty arrays so plotting/filtering won't break
        empty = np.array([])
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )


@pytest.fixture
def g():
    return DummyGalaxy()


def test_kT_and_E_cond_independence(g):
    # since use_kT_par=False, kT remains as in seed_parameters
    assert pytest.approx(50.0) == g.kT
    # E_cond not used here


def test_B_G_sigma_beta_monotonic(g):
    B = g.B_G()
    assert isinstance(B, float) and B > 0

    sigma = g.sigma_mag()
    assert isinstance(sigma, float) and sigma > 0

    beta = g.beta_mag()
    assert isinstance(beta, float) and beta > 0


def test_R_c_cm_and_E_cor(g):
    # R_c_cm = r_c * 1.5e5 * 10**log_M
    expected_Rc = g.r_c * 1.5e5 * 10**g.log_M
    assert np.isclose(g.R_c_cm(), expected_Rc)

    E_th, E_nt, E_B = g.E_cor()
    # check ordering and positivity
    assert E_B > 0 and E_nt > 0 and E_th > 0
    # E_B = eps_B * E_th, so E_th > E_B if eps_B<1
    eps_B = 10**g.log_eps_B
    assert np.isclose(E_B, eps_B * E_th)


def test_calculate_L_cor_and_integrals(g):
    nu = np.geomspace(1e9, 1e11, 20)
    # should set g.L_cor
    g.calculate_L_cor(nu)
    assert hasattr(g, "L_cor")
    arr = np.asarray(g.L_cor)
    assert arr.shape == nu.shape
    assert np.all(arr >= 0)

    # bolometric luminosity via trapezoid
    g.calculate_L_bol_cor(nu)
    expected = np.trapz(g.L_cor, nu)
    assert np.isclose(g.L_bol_cor, expected)

    # peak luminosity
    g.calculate_L_peak_cor(nu)
    idx = np.argmax(g.L_cor)
    nu_p = nu[idx]
    L_p = g.L_cor[idx]
    assert np.isclose(g.L_peak_cor, nu_p * L_p)


def test_Edd_ratio(g):
    L_X = 1e44
    # L_Edd = 1.26e38 * M_BH, M_BH = 10**log_M
    M_BH = 10**g.log_M
    expected = L_X / (1.26e38 * M_BH)
    assert np.isclose(g.calculate_Edd_ratio(L_X), expected)


def test_calculate_model_components_and_total(g):
    nu = np.array([1e10, 2e10, 5e10])
    # call with all keyword parameters
    total = g.calculate_model(
        nu,
        z=g.z,
        magnification=g.magnification,
        log_M=g.log_M,
        r_c=g.r_c,
        tau_T=g.tau_T,
        log_delta=g.log_delta,
        kT=g.kT,
        log_eps_B=g.log_eps_B,
        p=g.p,
        alpha_sy=g.alpha_sy,
        sy_scale=g.sy_scale,
        ff_scale=g.ff_scale,
        beta_RJ=g.beta_RJ,
        tau1_freq=g.tau1_freq,
        RJ_scale=g.RJ_scale,
        nu_tau1_cl=g.nu_tau1_cl,
        f_cov=g.f_cov,
        nu_tau1_diff=g.nu_tau1_diff,
    )
    # Should produce a numpy array of same shape
    assert isinstance(total, np.ndarray)
    assert total.shape == nu.shape
    # Total = sum of the four components
    comp_sum = g.S_cor + g.S_dust + g.S_ff + g.S_sync
    assert np.allclose(total, comp_sum)


def test_set_prior_bounds_structure(g):
    bounds = g.set_prior_bounds()
    expected_keys = {
        "z",
        "magnification",
        "log_M",
        "r_c",
        "tau_T",
        "log_delta",
        "kT",
        "log_eps_B",
        "p",
        "alpha_sy",
        "sy_scale",
        "ff_scale",
        "beta_RJ",
        "tau1_freq",
        "RJ_scale",
        "nu_tau1_cl",
        "f_cov",
        "nu_tau1_diff",
    }
    assert set(bounds.keys()) == expected_keys

    for key, (mn, mx) in bounds.items():
        # allow int or float
        assert isinstance(mn, (int, float))
        assert isinstance(mx, (int, float))
        assert mn < mx
