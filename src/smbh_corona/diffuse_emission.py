
#!/usr/bin/env python
# coding: utf-8

import numpy as np

from smbh_corona.constants import *


def pl_emission(nu_obs_frame, lg_scaling, alpha, z, nu_0=100e9):
    # nu_0 is a reference frequency
    nu_rest_frame = nu_obs_frame * (1+z)
    nu_0_rest_frame = nu_0 * (1+z)
    return 10**lg_scaling * (nu_rest_frame/nu_0_rest_frame)**(alpha)


def pl_emission_abs(nu_obs_frame, lg_scaling, alpha, z, nu_tau1_cl, nu_tau1_diff, nu_0=100e9):
    # f-f power-law emission that switches from optically thin to thick
    # nu_0 is a reference frequency
    # nu_tau1_cl is the (rest) frequency in GHz at which ionized clumps become opaque to the radio emission
    # nu_tau1_diff is the (rest) frequency in GHz at which the more dilute, diffuse medium absorbs radio emission
    nu_rest_frame = nu_obs_frame * (1+z)
    nu_0_rest_frame = nu_0 * (1+z)
    tau_nu = (nu_rest_frame/(nu_tau1_cl*1e9))**-2.1
    tau_nu_diff = (nu_rest_frame/(nu_tau1_diff*1e9))**-2.1
    return ( 10**lg_scaling * (nu_rest_frame/nu_0_rest_frame)**(alpha) * ( 1 - np.exp(-tau_nu))/tau_nu ) * np.exp(-tau_nu_diff)


def pl_emission_cl_abs(nu_obs_frame, lg_scaling, alpha, z, nu_tau1_cl, f_cov, nu_tau1_diff, nu_0=100e9):
    # power-law emission partially absorbed by a clumpy medium (e.g. Ramírez-Olivencia+2022)
    # nu_0 is a reference frequency to help the scaling
    # nu_tau1_cl is the (rest) frequency in GHz a which ionized clumps become opaque to the radio emission
    # nu_tau1_diff is the (rest) frequency in GHz at which the more dilute, diffuse medium absorbs radio emission
    # f_cov is the covering factor of the ionized medium
    # A more correct expression is in Eq. A3 from Ramirez-Olivencia+2022.
    nu_rest_frame = nu_obs_frame * (1+z)
    nu_0_rest_frame = nu_0 * (1+z)
    if f_cov > 0.01:
        tau_nu = (nu_rest_frame/(nu_tau1_cl*1e9))**-2.1
        cl_abs = 1 - f_cov * (1-np.exp(-tau_nu))
    else:
        cl_abs = 1.0
    tau_nu_diff = (nu_rest_frame/(nu_tau1_diff*1e9))**-2.1
    return ( 10**lg_scaling * (nu_rest_frame/nu_0_rest_frame)**(alpha) * cl_abs ) * np.exp(-tau_nu_diff)


def RJ_dust(nu_obs_frame, lg_scaling_rj, nu_tau1_rest, beta, z,
            T_d, nu_0=100e9, x_cut=0.001):
    """
    Modified blackbody (RJ approx. with Planck correction factor).

    Parameters
    ----------
    nu_obs_frame : array_like
        Observed-frame frequency (Hz).
    lg_scaling_rj : float
        log10 scaling factor for the RJ flux.
    nu_tau1_rest : float
        (user-original convention) frequency where tau=1. 
    beta : float
        dust emissivity index.
    z : float
        redshift.
    T_d : float
        dust temperature in K.
    nu_0 : float, optional
        reference frequency (Hz) in the observed frame (nu_0_rest_frame = nu_0*(1+z)).
    x_cut : float, optional
        threshold for h*nu/(k*T) below which we set the correction factor to 1

    Returns
    -------
    flux : array_like
    """
    # rest-frame frequencies
    nu_0_rest_frame = nu_0 * (1 + z)
    nu_rest_frame = nu_obs_frame * (1 + z)

    # optical depth 
    tau_nu = (nu_rest_frame / (nu_tau1_rest * 1e9)) ** beta

    # Planck->RJ correction factor: x/(exp(x)-1) where x = h*nu/(k*T)
    x = (h * nu_rest_frame) / (k_B * T_d)
    # use expm1 for numerical precision: exp(x)-1 = np.expm1(x)
    # set factor=1 where x is extremely small (RJ regime)
    factor = np.where(x > x_cut, x / np.expm1(x), 1.0)

    # final returned quantity 
    return 10**lg_scaling_rj * (1 - np.exp(-tau_nu)) * (nu_rest_frame / nu_0_rest_frame)**2 * factor