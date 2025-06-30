import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
import numpy as np
import pytest

from smbh_corona.plot_utils import (
    nice_fonts,
    plot_components,
    plot_components_confidence_intervals,
    set_latex_labels,
    plot_bands,
)


def test_nice_fonts_applied():
    """rcParams should have been updated at import time."""
    for key, val in nice_fonts.items():
        rc_val = matplotlib.rcParams[key]
        # font.family is stored as a list
        if key == "font.family":
            assert val in rc_val
        else:
            assert rc_val == val


def test_plot_components_with_and_without_legend():
    nu = np.array([1e9, 2e9, 3e9])
    comps = [np.linspace(i, i + 2, len(nu)) for i in range(5)]
    names = ["total", "corona", "dust", "f-f", "sync"]

    # with legend
    fig, ax = plt.subplots()
    plot_components(ax, nu, comps, label_prefix="foo_", show_legend=True, alpha=0.42)
    lines = ax.get_lines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert np.allclose(line.get_xdata(), nu / 1e9)
        assert pytest.approx(line.get_alpha()) == 0.42
        assert line.get_label() == f"foo_{names[i]}"
    plt.close(fig)

    # without legend → auto-generated labels start with "_"
    fig, ax = plt.subplots()
    plot_components(ax, nu, comps, label_prefix="", show_legend=False)
    lines = ax.get_lines()
    assert len(lines) == 5
    for line in lines:
        lbl = line.get_label()
        assert lbl.startswith("_")
    plt.close(fig)


@pytest.mark.parametrize("sigmas", [1, 2, 3])
def test_plot_components_confidence_intervals_runs_without_error(sigmas):
    """
    plot_components_confidence_intervals should fill_between 1×n
    (for σ=1), 2×n (for σ=2) or 3×n (for σ=3) times without error.
    """
    nu = np.linspace(1e9, 1e10, 50)
    # build dummy samples: 5 components, each with shape (n_samples, len(nu))
    comps_samples = [
        np.random.normal(loc=i, scale=0.1, size=(20, len(nu))) for i in range(5)
    ]

    fig, ax = plt.subplots()
    ax.clear()
    plot_components_confidence_intervals(ax, nu, comps_samples, sigmas=sigmas)

    # count how many PolyCollections were added
    pcs = [c for c in ax.get_children() if isinstance(c, PolyCollection)]
    # each component contributes exactly 'sigmas' fills
    assert len(pcs) == 5 * sigmas
    plt.close(fig)


def test_set_latex_labels():
    labels = set_latex_labels()
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
    assert set(labels) == expected_keys
    for latex in labels.values():
        assert latex.startswith("$") and latex.endswith("$")


def test_plot_bands():
    fig, ax = plt.subplots()
    bands = ["L", "C", "B3"]
    ymin, ymax = 1e-3, 1e1

    plot_bands(ax, ymin, ymax, bands)

    # one Rectangle per band
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == len(bands)

    # one axvline + one text per band
    vlines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    texts = ax.texts
    assert len(vlines) == len(bands)
    assert len(texts) == len(bands)

    plt.close(fig)
