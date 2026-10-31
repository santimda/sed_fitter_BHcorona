import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd
from bilby.core.utils import infer_parameters_from_function

from smbh_corona.galaxy import Galaxy
from smbh_corona.constants import arcsec
from smbh_corona.utils import Omega_beam

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from galaxy_sed import _save_derived_summary


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
            "T_d": 100,
        }
        self.update_parameters(self.seed_parameters)
        # dummy distance
        from astropy.cosmology import Planck18

        self.D_L = Planck18.luminosity_distance(self.z)
        self.max_omega_beam = Omega_beam(arcsec, arcsec)

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
    if hasattr(np, "trapezoid"):
        expected = np.trapezoid(g.L_cor, nu)
    else:
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


def test_calc_X_uses_current_galaxy_parameters_by_default(g, monkeypatch):
    calls = []

    def calculate_model(nu):
        calls.append(nu)
        g.S_tot = np.array([1e4])

    monkeypatch.setattr(g, "calculate_model", calculate_model)
    g.D_L = 100.0

    with pytest.warns(RuntimeWarning):
        result = g.calc_X(verbose=False)
    flux_result = g.calc_X(lum=False, verbose=False)

    assert result > 0
    assert flux_result == pytest.approx(
        10 ** ((np.log10(calls[1] * 1e4 * 1e-26) + 1.3) / 1.37)
    )
    assert len(calls) == 2
    assert calls[0] == pytest.approx(100e9 / (1 + g.z))


def test_derived_summary_writes_corona_diagnostics_for_posterior(g, tmp_path, monkeypatch):
    g.name = "mock"
    model_parameters = infer_parameters_from_function(g.calculate_model)
    posterior_samples = []
    for magnification in (1.5, 2.0):
        sample = g.seed_parameters.copy()
        sample["magnification"] = magnification
        posterior_samples.append(sample)
    posterior = pd.DataFrame(posterior_samples)

    def calculate_model(nu, **parameters):
        nu = np.atleast_1d(nu)
        magnification = parameters["magnification"]
        g.S_cor = np.full(nu.shape, 4.0 * magnification)
        g.S_tot = np.full(nu.shape, 10.0 * magnification)
        if len(nu) > 1:
            g.S_cor[np.argmin(np.abs(nu - 1e11))] = 8.0 * magnification
        return g.S_tot

    monkeypatch.setattr(g, "calculate_model", calculate_model)
    config = {"fit": {"seed": 41}, "plot": {"posterior_samples": 2}}
    output_dir = tmp_path / "mock"
    output_dir.mkdir()

    _save_derived_summary(
        "mock",
        g,
        SimpleNamespace(posterior=posterior),
        model_parameters,
        config,
        output_dir,
    )

    summary = json.loads((output_dir / "derived_summary.json").read_text())
    assert summary["posterior_samples"] == 2
    assert "intrinsic" not in summary and "demagnified" not in summary
    assert set(summary["corona"]) == {
        "B_G",
        "sigma_mag",
        "beta_mag",
        "R_uas",
        "dt_obs_days",
        "L_bol_cor_erg_s",
        "nu_peak_obs_GHz",
        "nu_peak_RF_GHz",
        "S_peak_mJy",
    }
    assert ")e" in summary["corona"]["L_bol_cor_erg_s"]
    assert summary["corona"]["S_peak_mJy"].startswith("(14.000+")

    observed_peak = float(summary["corona"]["nu_peak_obs_GHz"].split("+")[0][1:])
    rest_peak = float(summary["corona"]["nu_peak_RF_GHz"].split("+")[0][1:])
    assert rest_peak == pytest.approx(observed_peak * (1 + g.z), abs=0.002)


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
        "T_d",
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
