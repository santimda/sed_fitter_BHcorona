import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# My plot config
nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "font.size": 20,
    "axes.linewidth": 1.5,
    "axes.titlesize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.major.size": 5.5,
    "xtick.minor.size": 4,
    "ytick.major.size": 5.5,
    "ytick.minor.size": 3.5,
    "xtick.major.width": 1.4,
    "xtick.minor.width": 1.3,
    "ytick.major.width": 1.4,
    "ytick.minor.width": 1.3,
}

matplotlib.rcParams.update(nice_fonts)


def plot_components(ax, nu, components, label_prefix, show_legend=True, alpha=0.5):
    component_names = ["total", "corona", "dust", "f-f", "sync"]
    component_colors = ["b", "#5ebaf2", "violet", "orange", "grey"]
    component_lines = ["solid", "dashed", "dotted", "dotted", "dotted"]

    for i, component in enumerate(components):
        if show_legend:
            ax.plot(
                nu / 1e9,
                component,
                linestyle=component_lines[i],
                label=label_prefix + component_names[i],
                color=component_colors[i],
                alpha=alpha,
            )
        else:
            ax.plot(
                nu / 1e9,
                component,
                linestyle=component_lines[i],
                color=component_colors[i],
                alpha=alpha,
            )


def plot_components_confidence_intervals(ax, nu, components_samples, sigmas=1):
    component_colors = [
        "b",
        "#5ebaf2",
        "violet",
        "sandybrown",
        "grey",
    ]  # Match with plot_components

    for i, component_samples in enumerate(components_samples):
        # Calculate the confidence intervals of 1, 2 and/or 3 sigmas
        percentiles = [16, 84, 2.5, 97.5, 0.15, 99.85]
        lower_bounds = np.percentile(component_samples, percentiles[::2], axis=0)
        upper_bounds = np.percentile(component_samples, percentiles[1::2], axis=0)

        # Plot the filled curves for the confidence intervals
        labels = [r"$1\sigma$", r"$2\sigma$", r"$3\sigma$"]
        alphas = [0.37, 0.31, 0.24]

        if sigmas == 1:
            ax.fill_between(
                nu / 1e9,
                lower_bounds[0],
                upper_bounds[0],
                color=component_colors[i],
                alpha=alphas[1],
                edgecolor=None,
            )
        elif sigmas == 2:
            ax.fill_between(
                nu / 1e9,
                lower_bounds[0],
                upper_bounds[0],
                color=component_colors[i],
                alpha=alphas[1],
                edgecolor=None,
            )
            ax.fill_between(
                nu / 1e9,
                lower_bounds[1],
                upper_bounds[1],
                color=component_colors[i],
                alpha=alphas[2],
                edgecolor=None,
            )
        elif sigmas == 3:
            for lower, upper, label, alpha_ in zip(
                lower_bounds, upper_bounds, labels, alphas
            ):
                ax.fill_between(
                    nu / 1e9,
                    lower,
                    upper,
                    color=component_colors[i],
                    alpha=alpha_,
                    edgecolor=None,
                )
        else:
            "Wrong sigmas value, only 1, 2, 3 are allowed"


def set_latex_labels():
    labels = {
        "z": r"$z$",
        "magnification": r"$\mu$",
        "log_M": r"$\log{M_\mathrm{BH}\,[\mathrm{M}_\odot]}$",
        "r_c": r"$r_\mathrm{c}$",
        "tau_T": r"$\tau_\mathrm{T}$",
        "log_delta": r"$\log{\delta}$",
        "kT": r"$k\,T_\mathrm{c}\,[\mathrm{keV}]$",
        "log_eps_B": r"$\log{\epsilon_B}$",
        "p": r"$p$",
        "alpha_sy": r"$\alpha_\mathrm{sy}$",
        "sy_scale": r"$\log{A_\mathrm{sy}}$",
        "ff_scale": r"$\log{A_\mathrm{ff}}$",
        "beta_RJ": r"$\beta$",
        "tau1_freq": r"$\nu_{\tau_1}\,[\mathrm{GHz}]$",
        "RJ_scale": r"$\log{A_\mathrm{d}}$",
        "nu_tau1_cl": r"$\nu_\mathrm{cl}\,[\mathrm{GHz}]$",
        "f_cov": r"$f_\mathrm{cov}$",
        "nu_tau1_diff": r"$\nu_\mathrm{diff}\,[\mathrm{GHz}]$",
    }
    return labels


def plot_bands(ax, ymin, ymax, bands_to_plot, y_positions=None, colors=None):
    """Plot VLA and/or ALMA bands as rectangles"""
    bands_dict = {
        "SKA-Low": {"start": 0.05, "width": 0.35},
        "SKA-Mid": {"start": 0.35, "width": 15.4},
        "SKA-Mid(5b)": {"start": 8.3, "width": 15.4},
        "L": {"start": 1, "width": 1},
        "S": {"start": 2, "width": 2},
        "C": {"start": 4, "width": 4},
        "X": {"start": 8, "width": 4},
        "Ku": {"start": 12, "width": 6},
        "K": {"start": 18, "width": 8.5},
        "Ka": {"start": 26.5, "width": 13.5},
        "Q": {"start": 40, "width": 10},
        "ALMA-B1": {"start": 35, "width": 15},
        "B1": {"start": 35, "width": 15},
        "ALMA-B3": {"start": 84, "width": 32},
        "B3": {"start": 84, "width": 32},
        "B4": {"start": 125, "width": 38},
        "B5": {"start": 163, "width": 48},
        "B6": {"start": 211, "width": 64},
        "B7": {"start": 275, "width": 98},
        "B8": {"start": 385, "width": 115},
    }

    default_color = "grey"
    default_pos = np.sqrt(ymin * ymax)

    if y_positions is None:
        y_positions = {band: default_pos for band in bands_to_plot}
    else:
        y_positions = {band: y_positions[i] for i, band in enumerate(bands_to_plot)}

    if colors is None:
        colors = {band: default_color for band in bands_to_plot}
    else:
        colors = {band: colors[i] for i, band in enumerate(bands_to_plot)}

    for band_name in bands_to_plot:
        if band_name in bands_dict:
            band = bands_dict[band_name]
            band["pos"] = y_positions[band_name]
            band["color"] = colors[band_name]

            # Add the rectangle
            ax.add_patch(
                Rectangle(
                    (band["start"], ymin),
                    band["width"],
                    ymax - ymin,
                    edgecolor="none",
                    facecolor=band["color"],
                    alpha=0.1,
                    lw=0,
                )
            )

            # Add the vertical dashed line with the text
            ax.axvline(
                band["start"] + band["width"] / 2,
                ymin=0,
                ymax=1,
                color=band["color"],
                alpha=0.7,
                linestyle="--",
            )
            ax.text(
                band["start"] + band["width"] / 2 + 0.9,
                band["pos"],
                band_name,
                color=band["color"],
                ha="right",
                va="center",
                rotation="vertical",
                fontsize=16,
            )
