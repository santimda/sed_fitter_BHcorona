#!/usr/bin/env python
"""Plot the empirical millimeter/X-ray correlations and their scatter."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from smbh_corona.correlations import (
    K22_SCATTER_DEX,
    R23_SCATTER_DEX,
    V26a_LX2_to_L100,
    V26A_XRAY_SCATTER_DEX,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("validation_output"),
    )
    return parser.parse_args()


def plot_relation(ax, x, y, scatter, label, color):
    ax.plot(x, y, label=label, color=color)
    ax.fill_between(
        x,
        y - scatter,
        y + scatter,
        color=color,
        alpha=0.2,
        linewidth=0,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_lx2 = np.linspace(40, 45, 200)
    lx2 = 10**log_lx2
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_relation(
        ax,
        log_lx2,
        1.22 * log_lx2 - 13.9,
        R23_SCATTER_DEX,
        "R23: 2-10 keV",
        "C0",
    )
    plot_relation(
        ax,
        log_lx2,
        1.22 * log_lx2 - 14.4,
        R23_SCATTER_DEX,
        "R23: 14-150 keV",
        "C1",
    )
    plot_relation(
        ax,
        log_lx2,
        np.log10(V26a_LX2_to_L100(lx2)),
        V26A_XRAY_SCATTER_DEX,
        "V26a: 2-10 keV",
        "C2",
    )
    ax.set_xlabel(r"$\log_{10}(L_X\,[\mathrm{erg\ s^{-1}}])$")
    ax.set_ylabel(r"$\log_{10}(L_{100\,\mathrm{GHz}}\,[\mathrm{erg\ s^{-1}}])$")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(
        args.output_dir / "L100_vs_LX_correlations.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)

    log_lx = np.linspace(40, 45, 200)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_relation(
        ax,
        log_lx,
        1.08 * log_lx - 7.41,
        K22_SCATTER_DEX,
        "K22: 2-10 keV",
        "C0",
    )
    plot_relation(
        ax,
        log_lx,
        1.19 * log_lx - 12.78,
        K22_SCATTER_DEX,
        "K22: 14-150 keV",
        "C1",
    )
    ax.set_xlabel(r"$\log_{10}(L_X\,[\mathrm{erg\ s^{-1}}])$")
    ax.set_ylabel(r"$\log_{10}(L_{230\,\mathrm{GHz}}\,[\mathrm{erg\ s^{-1}}])$")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(
        args.output_dir / "L230_vs_LX_K22.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
