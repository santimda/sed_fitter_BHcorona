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


def test_corona_magnetic_diagnostics_are_positive(g):
    B = g.B_G()
    sigma = g.sigma_mag()
    beta = g.beta_mag()
    assert B > 0 and sigma > 0 and beta > 0


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
    assert summary["emission"]["total"]["L100"].startswith("(")
    assert ")e" in summary["emission"]["total"]["L100"]
    assert summary["emission"]["total"]["eddington_ratio"]["q50"] > 0
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

    total = g.calculate_model(nu)

    assert isinstance(total, np.ndarray)
    assert total.shape == nu.shape
    comp_sum = g.S_cor + g.S_dust + g.S_ff + g.S_sync
    assert np.allclose(total, comp_sum)
