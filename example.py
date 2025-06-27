#!/usr/bin/env python
# example.py
# Demonstration of how to use the sed_fitter_BHcorona.Galaxy class
# to compute and plot the millimeter SED (data + model).

import numpy as np

# make sure the package is installed (from the project root):
#    pip install -e .

from smbh_corona.galaxy import Galaxy, Omega_beam
from smbh_corona.constants import *

def main():
    # ------------------------------------------------------------------------------
    # 1) Create the Galaxy object. It will
    #      • load parameters from data/parameters/NGC985.par
    #      • load flux data from data/fluxes/NGC985_flux.dat
    #      • filter very-high-resolution or low resolution data (e.g, between 0.09"--1")
    # ------------------------------------------------------------------------------
    gal = Galaxy("NGC985")
    Omega_beam_min, Omega_beam_max = Omega_beam(0.09*arcsec, 0.09*arcsec), Omega_beam(0.3*arcsec, 0.3*arcsec)
    gal.filter_data_by_beam(Omega_beam_min=Omega_beam_min, Omega_beam_max=Omega_beam_max)
    
    # ------------------------------------------------------------------------------
    # 2) Pick a grid of observing frequencies [Hz]
    # ------------------------------------------------------------------------------
    nu = np.geomspace(1e9, 1e12, 200)

    # ------------------------------------------------------------------------------
    # 3) Compute the total model SED (returns a numpy array of mJy, shape == nu.shape)
    # ------------------------------------------------------------------------------
    sed_model = gal.calculate_model(nu)
    print(f"Computed model SED array of shape {sed_model.shape}")

    # ------------------------------------------------------------------------------
    # 4) Over-plot your model on top of the data, using Galaxy.plot_model()
    # ------------------------------------------------------------------------------
    gal.plot_model(nu)

    # ------------------------------------------------------------------------------
    # 5) Optionally, save the model curve for later analysis
    # ------------------------------------------------------------------------------
    out = np.vstack([nu, sed_model]).T
    np.savetxt("NGC985_model_sed.txt",
               out,
               header="nu_Hz    S_mJy",
               fmt="%e  %e")
    print("Wrote model to NGC985_model_sed.txt")

if __name__ == "__main__":
    main()
