#!/usr/bin/env python
"""Generate, fit, and verify deterministic test_galaxy mock data."""

import argparse
import json
import logging
from pathlib import Path
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import bilby

logging.getLogger("bilby").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"bilby\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"(bilby|emcee)\..*")

from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.core.utils import infer_parameters_from_function

from smbh_corona.galaxy import Galaxy
from smbh_corona.config import FLUX_PATH
from smbh_corona.plot_utils import (
    nice_fonts,
    plot_components,
    plot_components_confidence_intervals,
    set_latex_labels,
)
from smbh_corona.utils import GaussianLikelihood_withUL_andLL, generate_initial_positions


FIT_PARAMETERS = [
    "r_c",
    "log_delta",
    "alpha_sy",
    "sy_scale",
    "ff_scale",
    "RJ_scale",
    "T_d",
]


def configure_quiet_output():
    logging.getLogger("bilby").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"bilby\..*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"(bilby|emcee)\..*")

MOCK_FREQUENCIES_GHZ = np.array([
    0.8, 1.25, 1.75, 2.5, 3.5, 4.5, 7.5, 8.5, 11.5, 13.5, 17.0, 19.5,
    25.0, 28.5, 38.0, 42.5, 47.5, 38.75, 46.25, 75.0, 105.0, 90.0,
    110.0, 135.0, 155.0, 175.0, 200.0, 225.0, 260.0, 300.0, 350.0,
    410.0, 480.0, 630.0, 690.0, 1874.0, 2998.0, 4283.0, 9000.0,
])


def generate_mock_data():
    """Generate noiseless mock fluxes from the current test-galaxy truth."""
    sys_err = 0.03  # Assume 3% statistical flux uncertainty
    flux_path = FLUX_PATH / "test_galaxy_flux.dat"
    flux_path.parent.mkdir(parents=True, exist_ok=True)

    # Galaxy initialization requires a data file, so bootstrap it with the
    # fixed frequency grid before evaluating the truth model.
    placeholder = "\n".join(
        f"{frequency:.12g} 1.0 0.01 1.0 1.0 0"
        for frequency in MOCK_FREQUENCIES_GHZ
    )
    flux_path.write_text(
        "# Freq [GHz] Flux [mJy] Flux_err [mJy] theta1 [arcsec] theta2 [arcsec] Flag\n"
        + placeholder
        + "\n"
    )

    galaxy = Galaxy("test_galaxy", use_kT_par=False, use_E_cond=True)
    fluxes = galaxy.calculate_model(MOCK_FREQUENCIES_GHZ * 1e9)
    rows = [
        f"{frequency:.12g} {flux:.12g} {sys_err * flux:.12g} 1.0 1.0 0"
        for frequency, flux in zip(MOCK_FREQUENCIES_GHZ, fluxes)
    ]
    flux_path.write_text(
        "# Freq [GHz] Flux [mJy] Flux_err [mJy] theta1 [arcsec] theta2 [arcsec] Flag\n"
        + "\n".join(rows)
        + "\n"
    )
    return flux_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsteps", type=int, default=4000)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_test_output"))
    return parser.parse_args()


def build_priors(galaxy, model_parameters, fit_parameters):
    prior_bounds = galaxy.set_prior_bounds(
        delta_r_c=200.0,
        delta_log_delta=1.5,
    )
    labels = set_latex_labels()
    priors = PriorDict()
    for parameter in model_parameters:
        if parameter in fit_parameters:
            priors[parameter] = Uniform(
                *prior_bounds[parameter],
                name=parameter,
                latex_label=labels[parameter],
            )
        else:
            priors[parameter] = DeltaFunction(
                galaxy.seed_parameters[parameter],
                name=parameter,
                latex_label=labels[parameter],
            )
    return priors, prior_bounds


def fit_mock_galaxy(args):
    configure_quiet_output()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mock_data_path = generate_mock_data()
    galaxy = Galaxy("test_galaxy", use_kT_par=False, use_E_cond=True)
    model_parameters = infer_parameters_from_function(galaxy.calculate_model)
    fit_parameters = galaxy.get_fit_parameters(FIT_PARAMETERS)
    priors, prior_bounds = build_priors(galaxy, model_parameters, fit_parameters)

    x_data = galaxy.data_frequencies * 1e9
    y_data = galaxy.data_fluxes
    y_error = galaxy.data_fluxes_error
    likelihood = GaussianLikelihood_withUL_andLL(
        x_data,
        y_data,
        galaxy.calculate_model,
        sigma=y_error,
    )

    initial_pos = generate_initial_positions(
        prior=priors,
        fit_parameters=fit_parameters,
        prior_bounds=prior_bounds,
        seed_parameters=galaxy.seed_parameters,
        nwalkers=args.nwalkers,
    )

    results = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="emcee",
        pos0=initial_pos,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burn_in_fraction=0.3,
        nthreads=1,
        outdir=str(args.output_dir),
        label="test_galaxy",
        save=False,
        resume=False,
    )

    summary = {
        "seed": args.seed,
        "nwalkers": args.nwalkers,
        "nsteps": args.nsteps,
        "mock_data": str(mock_data_path),
        "fixed_parameters": {"log_M": galaxy.seed_parameters["log_M"]},
        "fit_parameters": fit_parameters,
        "prior_bounds": {key: list(prior_bounds[key]) for key in fit_parameters},
        "truth": {key: galaxy.seed_parameters[key] for key in fit_parameters},
        "posterior": {},
    }

    best_parameters = galaxy.seed_parameters.copy()
    for parameter in fit_parameters:
        values = np.asarray(results.posterior[parameter])
        lower, median, upper = np.percentile(values, [16, 50, 84])
        truth = summary["truth"][parameter]
        within_one_sigma = lower <= truth <= upper
        summary["posterior"][parameter] = {
            "lower": float(lower),
            "median": float(median),
            "upper": float(upper),
            "truth": float(truth),
            "within_one_sigma": bool(within_one_sigma),
        }
        best_parameters[parameter] = median
        status = "PASS" if within_one_sigma else "FAIL"
        print(f"{status} {parameter}: {median:.6g} [{lower:.6g}, {upper:.6g}] truth={truth:.6g}")

    summary["all_within_one_sigma"] = all(
        result["within_one_sigma"] for result in summary["posterior"].values()
    )
    print(f"all parameters within 1-sigma: {summary['all_within_one_sigma']}")

    summary_path = args.output_dir / "test_galaxy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    data_xmin = np.min(galaxy.data_frequencies)
    data_xmax = np.max(galaxy.data_frequencies)
    plot_frequencies = np.geomspace(0.5 * data_xmin * 1e9, 1.1 * data_xmax * 1e9, 250)
    galaxy.calculate_model(
        plot_frequencies,
        *[best_parameters[parameter] for parameter in model_parameters],
    )

    n_posterior_samples = min(1600, len(results.posterior))
    sampled_posterior = results.posterior[model_parameters].sample(
        n=n_posterior_samples,
        random_state=args.seed,
    )
    accepted_samples = sampled_posterior.to_dict("records")
    samples = galaxy.generate_all_y_samples(
        plot_frequencies,
        accepted_samples,
        model_parameters,
        plot_samples=False,
    )
    samples_components = [samples[index] for index in range(5)]

    matplotlib.rcParams.update(nice_fonts)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        galaxy.data_frequencies,
        galaxy.data_fluxes,
        yerr=galaxy.data_fluxes_error,
        fmt="o",
        color="black",
        markerfacecolor="none",
        label="data",
        markersize=4,
        elinewidth=1,
        zorder=12,
    )
    plot_components_confidence_intervals(
        ax,
        plot_frequencies,
        samples_components,
        sigmas=1,
    )
    plot_components(
        ax,
        plot_frequencies,
        [galaxy.S_tot, galaxy.S_cor, galaxy.S_dust, galaxy.S_ff, galaxy.S_sync],
        label_prefix="",
        show_legend=True,
        alpha=0.9,
    )
    ax.set_xlim(0.5 * data_xmin, 1.1 * data_xmax)
    ax.set_ylim(0.1 * np.min(galaxy.data_fluxes), 2.0 * np.max(galaxy.data_fluxes))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu_{\rm obs}$ [GHz]")
    ax.set_ylabel(r"$S(\nu)$ [mJy]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_galaxy_fit.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    corner = results.plot_corner(
        truth={parameter: summary["truth"][parameter] for parameter in fit_parameters},
        save=False,
    )
    corner.savefig(
        args.output_dir / "posterior_test_galaxy.pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(corner)

    if not summary["all_within_one_sigma"]:
        raise RuntimeError("Not all injected parameters are within the 1-sigma intervals")

    return summary


if __name__ == "__main__":
    fit_mock_galaxy(parse_args())
