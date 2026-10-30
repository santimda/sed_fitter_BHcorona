#!/usr/bin/env python
"""Fit and plot a galaxy SED using its JSON workflow configuration."""

import argparse
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.getLogger("bilby").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"bilby\..*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"(bilby|emcee)\..*"
)

from bilby import run_sampler
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.core.result import read_in_result
from bilby.core.utils import infer_parameters_from_function

from smbh_corona.config import CONFIGS_PATH
from smbh_corona.correlations import (
    L100_to_Lbol,
    V26a_L100_to_LX2,
    V26A_BOLOMETRIC_SCATTER_DEX,
    V26A_XRAY_SCATTER_DEX,
    flux_to_nu_luminosity,
)
from smbh_corona.galaxy import Galaxy
from smbh_corona.plot_utils import (
    nice_fonts,
    plot_components,
    plot_components_confidence_intervals,
    set_latex_labels,
)
from smbh_corona.utils import (
    GaussianLikelihood_withUL_andLL,
    generate_initial_positions,
    reduced_chi_square,
)


def parse_bool(value):
    value = value.lower()
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected True or False")
    return value == "true"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("galaxy_name", nargs="?", default="NGC985")
    parser.add_argument("--fit", type=parse_bool, default=True)
    return parser


def load_config(galaxy_name, configs_path=CONFIGS_PATH):
    """Load a galaxy config, creating it from the default template if absent."""
    configs_path = Path(configs_path)
    default_path = configs_path / "default_config.json"
    config_path = configs_path / f"{galaxy_name}_config.json"
    if not config_path.exists():
        if not default_path.exists():
            raise FileNotFoundError(f"Default configuration not found: {default_path}")
        configs_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_path, config_path)
        print(f"No configuration found for {galaxy_name}.")
        print(f"Created {config_path} from the default template.")
        print("Review this file before using the result for scientific analysis.")

    with config_path.open() as config_file:
        config = json.load(config_file)
    _validate_config(config, config_path)
    return config, config_path


def _validate_config(config, config_path):
    required = {
        "model": ("use_kT_par", "use_E_cond", "eta_mag", "xi_ep"),
        "data": ("min_ang_scale_arcsec", "max_ang_scale_arcsec"),
        "fit": (
            "parameters",
            "nsteps",
            "nwalkers",
            "seed",
            "w",
            "burn_in_fraction",
            "perturb_fraction",
            "prior_options",
        ),
        "plot": (
            "xmin_ghz",
            "xmax_ghz",
            "ymin_mjy",
            "ymax_mjy",
            "posterior_samples",
            "confidence_sigmas",
        ),
        "output": ("root",),
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"Missing '{section}' section in {config_path}")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(
                f"Missing keys in {config_path} section '{section}': {missing}"
            )


def _configure_quiet_output():
    logging.getLogger("bilby").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"bilby\..*")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module=r"(bilby|emcee)\..*"
    )


def _make_galaxy(galaxy_name, config):
    model_config = config["model"]
    data_config = config["data"]
    galaxy = Galaxy(
        galaxy_name,
        use_kT_par=model_config["use_kT_par"],
        use_E_cond=model_config["use_E_cond"],
        eta_mag=model_config["eta_mag"],
        xi_ep=model_config["xi_ep"],
        min_ang_scale=data_config["min_ang_scale_arcsec"],
        max_ang_scale=data_config["max_ang_scale_arcsec"],
    )
    galaxy.filter_data_by_beam()
    return galaxy


def _output_dir(galaxy_name, config):
    return Path(config["output"]["root"]) / galaxy_name


def _prepare_output(galaxy_name, config):
    output_dir = _output_dir(galaxy_name, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "emcee_fit", ignore_errors=True)
    for filename in (
        "fit_result.json",
        "fit_summary.json",
        "run_metadata.json",
        f"posterior_{galaxy_name}.pdf",
        f"SED_{galaxy_name}.pdf",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    return output_dir


def _make_priors(galaxy, model_parameters, fit_parameters, config):
    prior_bounds = galaxy.set_prior_bounds(**config["fit"]["prior_options"])
    latex_labels = set_latex_labels()
    priors = PriorDict()
    for parameter in model_parameters:
        if parameter in fit_parameters:
            priors[parameter] = Uniform(
                *prior_bounds[parameter],
                parameter,
                latex_label=latex_labels[parameter],
            )
        else:
            priors[parameter] = DeltaFunction(
                galaxy.seed_parameters[parameter],
                parameter,
                latex_label=latex_labels[parameter],
            )
    return priors, prior_bounds


def _limit_kwargs(galaxy):
    kwargs = {}
    if galaxy.data_frequencies_UL.size:
        kwargs.update(
            x_ul=galaxy.data_frequencies_UL * 1e9,
            y_ul=galaxy.data_fluxes_UL,
            sigma_ul=galaxy.data_fluxes_UL_error,
        )
    if galaxy.data_frequencies_LL.size:
        kwargs.update(
            x_ll=galaxy.data_frequencies_LL * 1e9,
            y_ll=galaxy.data_fluxes_LL,
            sigma_ll=galaxy.data_fluxes_LL_error,
        )
    return kwargs


def _round_for_json(value, decimals=3):
    if isinstance(value, dict):
        return {key: _round_for_json(item, decimals) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_for_json(item, decimals) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), decimals)
    return value


def _fit_summary(galaxy, results, model_parameters, fit_parameters, config):
    posterior = results.posterior
    median_parameters = {
        parameter: float(posterior[parameter].median())
        for parameter in model_parameters
    }
    galaxy.calculate_model(galaxy.data_frequencies * 1e9, **median_parameters)
    fit_config = config["fit"]
    return {
        "galaxy": galaxy.name,
        "seed": fit_config["seed"],
        "w": fit_config["w"],
        "nwalkers": fit_config["nwalkers"],
        "nsteps": fit_config["nsteps"],
        "fit_parameters": fit_parameters,
        "median_parameters": median_parameters,
        "posterior_quantiles": {
            parameter: {
                "q16": float(posterior[parameter].quantile(0.16)),
                "q50": float(posterior[parameter].quantile(0.50)),
                "q84": float(posterior[parameter].quantile(0.84)),
            }
            for parameter in fit_parameters
        },
        "reduced_chi_square": float(
            reduced_chi_square(
                galaxy.data_fluxes,
                galaxy.S_tot,
                galaxy.data_fluxes_error,
                len(fit_parameters),
            )
        ),
    }


def _save_run_metadata(galaxy, fit_parameters, config, config_path, output_dir):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    package_versions = {}
    for package in ("numpy", "scipy", "astropy", "bilby", "matplotlib"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    metadata = {
        "git_commit": commit or "unknown",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
        "working_directory": str(Path.cwd()),
        "output_directory": str(output_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config": config,
        "galaxy": galaxy.name,
        "fit_parameters": fit_parameters,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def fit_sed(galaxy_name, config, config_path):
    """Fit a galaxy SED and save the result, summary, chain, and metadata."""
    _configure_quiet_output()
    fit_config = config["fit"]
    np.random.seed(fit_config["seed"])
    random.seed(fit_config["seed"])
    output_dir = _prepare_output(galaxy_name, config)
    galaxy = _make_galaxy(galaxy_name, config)
    model_parameters = infer_parameters_from_function(galaxy.calculate_model)
    fit_parameters = galaxy.get_fit_parameters(fit_config["parameters"])
    priors, prior_bounds = _make_priors(
        galaxy, model_parameters, fit_parameters, config
    )
    initial_positions = generate_initial_positions(
        prior=priors,
        fit_parameters=fit_parameters,
        prior_bounds=prior_bounds,
        seed_parameters=galaxy.seed_parameters,
        nwalkers=fit_config["nwalkers"],
        use_broad_sampling=False,
        perturb_fraction=fit_config["perturb_fraction"],
    )
    likelihood = GaussianLikelihood_withUL_andLL(
        x=galaxy.data_frequencies * 1e9,
        y=galaxy.data_fluxes,
        func=galaxy.calculate_model,
        sigma=galaxy.data_fluxes_error,
        w=fit_config["w"],
        **_limit_kwargs(galaxy),
    )
    results = run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="emcee",
        pos0=initial_positions,
        nwalkers=fit_config["nwalkers"],
        nsteps=fit_config["nsteps"],
        burn_in_fraction=fit_config["burn_in_fraction"],
        outdir=str(output_dir),
        label="fit",
        save=True,
    )
    summary = _fit_summary(
        galaxy, results, model_parameters, fit_parameters, config
    )
    (output_dir / "fit_summary.json").write_text(
        json.dumps(_round_for_json(summary), indent=2) + "\n"
    )
    _save_run_metadata(
        galaxy, fit_parameters, config, config_path, output_dir
    )
    _save_corner_plot(
        results,
        fit_parameters,
        output_dir / f"posterior_{galaxy_name}.pdf",
    )
    _save_derived_summary(
        galaxy_name, galaxy, results, model_parameters, config, output_dir
    )
    return summary


def _posterior_components(
    galaxy, posterior, model_parameters, frequencies, sample_count, rng
):
    sample_count = min(sample_count, len(posterior))
    indices = rng.choice(len(posterior), size=sample_count, replace=False)
    components = [[] for _ in range(5)]
    for index in indices:
        parameters = posterior.iloc[index][model_parameters].to_dict()
        galaxy.calculate_model(frequencies, **parameters)
        for values, attribute in zip(
            components, ("S_tot", "S_cor", "S_dust", "S_ff", "S_sync")
        ):
            values.append(getattr(galaxy, attribute).copy())
    return [np.asarray(values) for values in components]


def _save_corner_plot(results, fit_parameters, output_path):
    medians = {
        parameter: float(results.posterior[parameter].median())
        for parameter in fit_parameters
    }
    figure = results.plot_corner(truth=medians, save=False)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _save_sed_plot(galaxy, results, model_parameters, config, output_path):
    plot_config = config["plot"]
    xmin, xmax = plot_config["xmin_ghz"], plot_config["xmax_ghz"]
    ymin, ymax = plot_config["ymin_mjy"], plot_config["ymax_mjy"]
    frequencies = np.geomspace(0.9 * xmin * 1e9, 1.1 * xmax * 1e9, 250)
    posterior = results.posterior
    rng = np.random.default_rng(config["fit"]["seed"])
    component_samples = _posterior_components(
        galaxy,
        posterior,
        model_parameters,
        frequencies,
        plot_config["posterior_samples"],
        rng,
    )
    median_parameters = {
        parameter: float(posterior[parameter].median())
        for parameter in model_parameters
    }
    galaxy.calculate_model(frequencies, **median_parameters)
    best_fit_components = [
        galaxy.S_tot.copy(),
        galaxy.S_cor.copy(),
        galaxy.S_dust.copy(),
        galaxy.S_ff.copy(),
        galaxy.S_sync.copy(),
    ]
    galaxy.calculate_model(galaxy.data_frequencies * 1e9, **median_parameters)
    model_at_data = galaxy.S_tot.copy()

    matplotlib.rcParams.update(nice_fonts)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_xlabel(r"$\nu_{\rm obs}$ [GHz]")
    axis.set_ylabel(r"$S(\nu)$ [mJy]")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.loglog()
    detection_mask = galaxy.data_type == 0
    axis.errorbar(
        galaxy.data_frequencies[detection_mask],
        galaxy.data_fluxes[detection_mask],
        yerr=galaxy.data_fluxes_error[detection_mask],
        fmt="o",
        markerfacecolor="none",
        color="black",
        alpha=1.0,
        markersize=4,
        elinewidth=1,
        zorder=12,
        label="data",
    )
    if np.any(galaxy.data_type == 2):
        fake_mask = galaxy.data_type == 2
        axis.errorbar(
            galaxy.data_frequencies[fake_mask],
            galaxy.data_fluxes[fake_mask],
            yerr=galaxy.data_fluxes_error[fake_mask],
            fmt="o",
            markerfacecolor="none",
            color="green",
            alpha=0.9,
            label="sim data",
            markersize=4,
            elinewidth=2,
            zorder=12,
        )
    if galaxy.data_frequencies_UL.size:
        axis.errorbar(
            galaxy.data_frequencies_UL,
            galaxy.data_fluxes_UL,
            yerr=0.12 * galaxy.data_fluxes_UL,
            uplims=True,
            fmt="_",
            color="#900000",
            alpha=0.9,
            markersize=3,
            label="_nolegend_",
        )
    if galaxy.data_frequencies_LL.size:
        axis.errorbar(
            galaxy.data_frequencies_LL,
            galaxy.data_fluxes_LL,
            yerr=0.12 * galaxy.data_fluxes_LL,
            lolims=True,
            fmt="_",
            color="darkgreen",
            alpha=0.9,
            markersize=3,
            label="_nolegend_",
        )

    plot_components_confidence_intervals(
        axis,
        frequencies,
        component_samples,
        sigmas=plot_config["confidence_sigmas"],
    )
    plot_components(
        axis,
        frequencies,
        best_fit_components,
        label_prefix="",
        show_legend=True,
        alpha=1,
    )
    axis.annotate(
        galaxy.name,
        xy=(0.04, 0.6),
        xytext=(0.5, 0.85),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=24,
    )
    axis.tick_params(direction="in", which="both", right=True, top=True)
    axis.legend(loc="upper left", ncol=2, columnspacing=0.8, fontsize=16)

    residuals_axis = make_axes_locatable(axis).append_axes(
        "bottom", size="25%", pad=0
    )
    residuals_axis.set_xscale("log")
    residuals_axis.set_ylabel(r"$\Delta$", fontsize=nice_fonts["axes.labelsize"])
    for sigma in (1, 2):
        residuals_axis.add_patch(
            Rectangle(
                (xmin, -sigma),
                xmax - xmin,
                2 * sigma,
                linewidth=0,
                edgecolor="none",
                facecolor="pink",
                alpha=0.25,
                zorder=1,
            )
        )
    residuals_axis.axhline(
        y=0, color="darkgrey", linestyle="--", linewidth=1, zorder=2
    )
    residuals = (galaxy.data_fluxes - model_at_data) / galaxy.data_fluxes_error
    residuals_axis.scatter(
        galaxy.data_frequencies,
        residuals,
        color="purple",
        marker="^",
        alpha=0.9,
        s=12,
        zorder=3,
    )
    residuals_axis.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    residuals_axis.set_yticklabels(["", "", "", -2, "", 0, "", 2, "", "", ""])
    residuals_axis.set_xlim(axis.get_xlim())
    residuals_axis.set_ylim(-4, 4)
    residuals_axis.tick_params(direction="in", which="both", right=True, top=False)
    figure.text(
        0.5,
        0.04,
        r"$\nu_{\rm obs}$ [GHz]",
        ha="center",
        va="center",
        fontsize=nice_fonts["axes.labelsize"],
    )
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def _summary_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "q16": float(np.percentile(values, 16)),
        "q50": float(np.percentile(values, 50)),
        "q84": float(np.percentile(values, 84)),
    }


def _derived_quantity_summary(samples, quantity):
    summary = _summary_stats([sample[quantity] for sample in samples])
    if quantity in ("L100", "L2_10", "Lbol"):
        return _format_luminosity_summary(summary)
    return summary


def _format_luminosity_summary(summary):
    median = summary["q50"]
    _, exponent = f"{median:.2e}".split("e")
    exponent = int(exponent)
    scale = 10**exponent
    return (
        f"({median / scale:.2f}+{(summary['q84'] - median) / scale:.2f}"
        f"-{(median - summary['q16']) / scale:.2f})e{exponent}"
    )


def _save_derived_summary(galaxy_name, galaxy, results, model_parameters, config, output_dir):
    posterior = results.posterior
    plot_config = config["plot"]
    sample_count = min(plot_config["posterior_samples"], len(posterior))
    rng = np.random.default_rng(config["fit"]["seed"])
    indices = rng.choice(len(posterior), size=sample_count, replace=False)
    derived = {"total": [], "corona_only": []}

    for index in indices:
        sample = posterior.iloc[index]
        parameters = sample[model_parameters].to_dict()
        z = float(sample["z"])
        magnification = float(sample["magnification"])
        distance_mpc = float(Planck18.luminosity_distance(z).value)
        observed_frequency_ghz = 100.0 / (1.0 + z)
        galaxy.calculate_model(observed_frequency_ghz * 1e9, **parameters)
        for component, attribute in (("total", "S_tot"), ("corona_only", "S_cor")):
            l100 = float(
                flux_to_nu_luminosity(
                    getattr(galaxy, attribute)[0],
                    distance_mpc,
                    100.0,
                    magnification=magnification,
                )
            )
            lx2 = float(V26a_L100_to_LX2(l100))
            lbol = float(L100_to_Lbol(l100))
            lx2 *= 10 ** rng.normal(0.0, V26A_XRAY_SCATTER_DEX)
            lbol *= 10 ** rng.normal(0.0, V26A_BOLOMETRIC_SCATTER_DEX)
            eddington_ratio = lx2 / (1.26e38 * 10 ** float(sample["log_M"]))
            derived[component].append(
                {
                    "L100": l100,
                    "L2_10": lx2,
                    "Lbol": lbol,
                    "eddington_ratio": eddington_ratio,
                }
            )

    summary = {
        "galaxy": galaxy_name,
        "relation": "Venselaar et al. 2026a",
        "rest_frequency_ghz": 100.0,
        "intrinsic": True,
        "demagnified": True,
        "scatter_applied": {
            "L2_10_dex": V26A_XRAY_SCATTER_DEX,
            "Lbol_dex": V26A_BOLOMETRIC_SCATTER_DEX,
        },
        "posterior_samples": sample_count,
        "emission": {
            component: {
                quantity: _derived_quantity_summary(samples, quantity)
                for quantity in ("L100", "L2_10", "Lbol", "eddington_ratio")
            }
            for component, samples in derived.items()
        },
    }
    (output_dir / "derived_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def plot_sed(galaxy_name, config):
    """Load a saved fit and regenerate its corner and SED plots."""
    _configure_quiet_output()
    output_dir = _output_dir(galaxy_name, config)
    results = read_in_result(outdir=str(output_dir), label="fit")
    galaxy = _make_galaxy(galaxy_name, config)
    model_parameters = infer_parameters_from_function(galaxy.calculate_model)
    _save_sed_plot(
        galaxy,
        results,
        model_parameters,
        config,
        output_dir / f"SED_{galaxy_name}.pdf",
    )


def main():
    args = build_parser().parse_args()
    config, config_path = load_config(args.galaxy_name)
    if args.fit:
        print(f"Fitting SED of {args.galaxy_name}")
        fit_sed(args.galaxy_name, config, config_path)
        summary_path = (_output_dir(args.galaxy_name, config) / "derived_summary.json").resolve()
        print(f"Summary of derived parameters saved to {summary_path}")
    print(f"Plotting SED of {args.galaxy_name}")
    plot_sed(args.galaxy_name, config)


if __name__ == "__main__":
    main()
