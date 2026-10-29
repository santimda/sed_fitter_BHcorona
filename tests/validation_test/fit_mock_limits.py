#!/usr/bin/env python
"""Fit a mixed detection, lower-limit, and upper-limit mock data set."""

import argparse
import copy
import json
from pathlib import Path

import bilby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smbh_corona.config import FLUX_PATH
from smbh_corona.galaxy import Galaxy
from smbh_corona.utils import GaussianLikelihood_withUL_andLL, generate_initial_positions

from fit_mock_galaxy import (
    FIT_PARAMETERS,
    MOCK_FREQUENCIES_GHZ,
    build_priors,
    cleanup_sampler_output,
    configure_quiet_output,
    validation_plot_style,
)
from bilby.core.utils import infer_parameters_from_function
from smbh_corona.plot_utils import plot_components, plot_components_confidence_intervals

MOCK_FREQUENCIES_GHZ = np.array([
    0.9, 1.5, 3.0, 8.0, 15.0, 26.0, 42.0, 80.0, 105.0, 
    135.0, 155.0, 175.0, 200.0, 225.0, 260.0, 300.0, 
    450.0, 650.0, 1874.0, 2998.0, 4283.0
])

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsteps", type=int, default=2500)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--w", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_test_output"))
    return parser.parse_args()


def generate_mixed_mock_data(frequencies=MOCK_FREQUENCIES_GHZ):
    """Write detections from 100--300 GHz and paired limits elsewhere."""
    frequencies = np.asarray(frequencies, dtype=float)
    flux_path = FLUX_PATH / "test_galaxy_flux.dat"
    flux_path.parent.mkdir(parents=True, exist_ok=True)
    detections = (frequencies >= 120.0) & (frequencies <= 300.0)

    placeholder_rows = []
    for frequency, is_detection in zip(frequencies, detections):
        if is_detection:
            placeholder_rows.append(f"{frequency:.12g} 1.0 0.01 1.0 1.0 0")
        else:
            placeholder_rows.extend([
                f"{frequency:.12g} 1.0 0.01 1.0 1.0 1",
                f"{frequency:.12g} 1.0 0.01 1.0 1.0 4",
            ])
    header = "# Freq [GHz] Flux [mJy] Flux_err [mJy] theta1 [arcsec] theta2 [arcsec] Flag\n"
    flux_path.write_text(header + "\n".join(placeholder_rows) + "\n")

    galaxy = Galaxy("test_galaxy", use_kT_par=False, use_E_cond=True)
    truth_fluxes = galaxy.calculate_model(frequencies * 1e9)
    rows = []
    for frequency, truth_flux, is_detection in zip(frequencies, truth_fluxes, detections):
        if is_detection:
            rows.append(f"{frequency:.12g} {truth_flux:.12g} {0.03 * truth_flux:.12g} 1.0 1.0 0")
        else:
            lower_limit = 0.85 * truth_flux
            upper_limit = 1.15 * truth_flux
            rows.extend([
                f"{frequency:.12g} {upper_limit:.12g} {0.03 * upper_limit:.12g} 1.0 1.0 1",
                f"{frequency:.12g} {lower_limit:.12g} {0.03 * lower_limit:.12g} 1.0 1.0 4",
            ])
    flux_path.write_text(header + "\n".join(rows) + "\n")
    return flux_path


def unique_limit_points(frequencies, fluxes, errors):
    """Keep one plotted limit per observing band."""
    _, indices = np.unique(frequencies, return_index=True)
    indices = np.sort(indices)
    return frequencies[indices], fluxes[indices], errors[indices]


def fit_mock_limits(args):
    configure_quiet_output()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mock_data_path = generate_mixed_mock_data(MOCK_FREQUENCIES_GHZ)

    galaxy = Galaxy("test_galaxy", use_kT_par=False, use_E_cond=True)
    model_parameters = infer_parameters_from_function(galaxy.calculate_model)
    fit_parameters = galaxy.get_fit_parameters(FIT_PARAMETERS)
    priors, prior_bounds = build_priors(galaxy, model_parameters, fit_parameters)

    likelihood = GaussianLikelihood_withUL_andLL(
        galaxy.data_frequencies * 1e9,
        galaxy.data_fluxes,
        galaxy.calculate_model,
        sigma=galaxy.data_fluxes_error,
        x_ul=galaxy.data_frequencies_UL * 1e9,
        y_ul=galaxy.data_fluxes_UL,
        sigma_ul=galaxy.data_fluxes_UL_error,
        x_ll=galaxy.data_frequencies_LL * 1e9,
        y_ll=galaxy.data_fluxes_LL,
        sigma_ll=galaxy.data_fluxes_LL_error,
        w=args.w,
    )
    initial_pos = generate_initial_positions(
        prior=priors,
        fit_parameters=fit_parameters,
        prior_bounds=prior_bounds,
        seed_parameters=galaxy.seed_parameters,
        nwalkers=args.nwalkers,
    )
    label = f"test_galaxy_limits_w{args.w:g}"
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
        label=label,
        save=False,
        resume=False,
    )

    summary = {
        "seed": args.seed,
        "nwalkers": args.nwalkers,
        "nsteps": args.nsteps,
        "w": args.w,
        "mock_data": str(mock_data_path),
        "fixed_parameters": {"log_M": galaxy.seed_parameters["log_M"]},
        "fit_parameters": fit_parameters,
        "posterior": {},
    }
    best_parameters = galaxy.seed_parameters.copy()
    for parameter in fit_parameters:
        values = np.asarray(results.posterior[parameter])
        lower, median, upper = np.percentile(values, [16, 50, 84])
        truth = galaxy.seed_parameters[parameter]
        within_one_sigma = lower <= truth <= upper
        summary["posterior"][parameter] = {
            "lower": float(lower),
            "median": float(median),
            "upper": float(upper),
            "truth": float(truth),
            "within_one_sigma": bool(within_one_sigma),
        }
        best_parameters[parameter] = median

    model_values = [best_parameters[parameter] for parameter in model_parameters]
    lower_prediction = galaxy.calculate_model(galaxy.data_frequencies_LL * 1e9, *model_values)
    upper_prediction = galaxy.calculate_model(galaxy.data_frequencies_UL * 1e9, *model_values)
    summary["limits"] = {
        "lower_within_bounds": bool(np.all(lower_prediction >= galaxy.data_fluxes_LL)),
        "upper_within_bounds": bool(np.all(upper_prediction <= galaxy.data_fluxes_UL)),
        "lower_prediction_min_margin": float(np.min(lower_prediction - galaxy.data_fluxes_LL)),
        "upper_prediction_min_margin": float(np.min(galaxy.data_fluxes_UL - upper_prediction)),
    }
    summary["all_within_one_sigma"] = all(
        item["within_one_sigma"] for item in summary["posterior"].values()
    )

    plot_frequencies = np.geomspace(
        0.5 * np.min(np.concatenate([
            galaxy.data_frequencies,
            galaxy.data_frequencies_UL,
            galaxy.data_frequencies_LL,
        ])) * 1e9,
        1.1 * np.max(np.concatenate([
            galaxy.data_frequencies,
            galaxy.data_frequencies_UL,
            galaxy.data_frequencies_LL,
        ])) * 1e9,
        250,
    )
    n_posterior_samples = min(1600, len(results.posterior))
    sampled_posterior = results.posterior[model_parameters].sample(
        n=n_posterior_samples,
        random_state=args.seed,
    )
    samples = galaxy.generate_all_y_samples(
        plot_frequencies,
        sampled_posterior.to_dict("records"),
        model_parameters,
        plot_samples=False,
    )
    galaxy.calculate_model(plot_frequencies, *model_values)
    ul_frequencies, ul_fluxes, _ = unique_limit_points(
        galaxy.data_frequencies_UL,
        galaxy.data_fluxes_UL,
        galaxy.data_fluxes_UL_error,
    )
    ll_frequencies, ll_fluxes, _ = unique_limit_points(
        galaxy.data_frequencies_LL,
        galaxy.data_fluxes_LL,
        galaxy.data_fluxes_LL_error,
    )
    matplotlib.rcParams.update(validation_plot_style())
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_components_confidence_intervals(
        ax,
        plot_frequencies,
        [samples[index] for index in range(5)],
        sigmas=3,
    )
    plot_components(
        ax,
        plot_frequencies,
        [galaxy.S_tot, galaxy.S_cor, galaxy.S_dust, galaxy.S_ff, galaxy.S_sync],
        label_prefix="",
        show_legend=True,
        alpha=0.9,
    )
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
    ax.errorbar(
        ul_frequencies,
        ul_fluxes,
        yerr=0.12 * ul_fluxes,
        uplims=True,
        fmt="_",
        color="#900000",
        alpha=0.9,
        markersize=3,
        label="_nolegend_",
    )
    ax.errorbar(
        ll_frequencies,
        ll_fluxes,
        yerr=0.12 * ll_fluxes,
        lolims=True,
        fmt="_",
        color="darkgreen",
        alpha=0.9,
        markersize=3,
        label="_nolegend_",
    )
    all_frequencies = np.concatenate([
        galaxy.data_frequencies,
        galaxy.data_frequencies_UL,
        galaxy.data_frequencies_LL,
    ])
    all_fluxes = np.concatenate([
        galaxy.data_fluxes,
        galaxy.data_fluxes_UL,
        galaxy.data_fluxes_LL,
    ])
    ax.set_xlim(0.5 * np.min(all_frequencies), 1.1 * np.max(all_frequencies))
    ax.set_ylim(0.1 * np.min(all_fluxes), 2.0 * np.max(all_fluxes))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu_{\rm obs}$ [GHz]")
    ax.set_ylabel(r"$S(\nu)$ [mJy]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{label}_fit.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    median_dic = {}
    for parameter in fit_parameters:
        median = results.get_one_dimensional_median_and_error_bar(
            parameter,
            fmt=".3f",
            quantiles=(0.16, 0.84),
        )
        median_dic[parameter] = median.median
    corner = results.plot_corner(truth=median_dic, save=False)
    corner.savefig(
        args.output_dir / f"posterior_{label}.pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(corner)

    summary_path = args.output_dir / f"{label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"w={args.w:g}: all parameters within 1-sigma: {summary['all_within_one_sigma']}")
    print(f"limit summary: {summary['limits']}")
    cleanup_sampler_output(args.output_dir, label)
    if not summary["all_within_one_sigma"]:
        raise RuntimeError("Not all injected parameters are within the 1-sigma intervals")
    return summary


def run_validations(args):
    weights = (1.0, 100.0) if args.w is None else (args.w,)
    try:
        summaries = []
        for weight in weights:
            summaries.append(fit_mock_limits(_with_w(args, weight)))
        return summaries
    finally:
        from fit_mock_galaxy import generate_mock_data

        generate_mock_data()


def _with_w(args, weight):
    updated_args = copy.copy(args)
    updated_args.w = weight
    return updated_args


if __name__ == "__main__":
    run_validations(parse_args())
