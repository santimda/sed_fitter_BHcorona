#!/usr/bin/env python
# example.py
# Demonstration of how to use the sed_fitter_BHcorona.Galaxy class
# to compute and plot the millimeter SED (data + model).

import numpy as np

# make sure the package is installed (from the project root):
#    pip install -e .

from smbh_corona.galaxy import Galaxy

def main():
    # ------------------------------------------------------------------------------
    # 1) Create the Galaxy object. It will
    #      • load parameters from data/parameters/NGC985.par
    #      • load flux data from data/fluxes/NGC985_flux.dat
    # ------------------------------------------------------------------------------
    gal = Galaxy("NGC985")

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
