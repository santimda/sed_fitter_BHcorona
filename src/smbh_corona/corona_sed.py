#!/usr/bin/env python
# coding: utf-8

import numpy as np

from smbh_corona.constants import *
from smbh_corona.thermalsyn import *
from astropy.cosmology import Planck18


def t_ff(r_c, M_BH):
    '''Free-fall time (Eq. 2 from Inoue+2019)
    r_c = normalized coronal radius
    M = BH mass in solar masses'''
    return 2.5 * 1e5 * np.sqrt(r_c/40) * (M_BH/1e8)

def calculate_Lnu(nu, M_BH, T, r_c, tau_T, eps_B, p, delta):
    '''Calculate the corona luminosity at frequencies "nu".
    Output: luminosity in erg/s/Hz.

    Parameters:
    -----------
    nu: np.array with frequencies (in the source reference frame)
    M_BH: BH mass in units of 10^8 M_sun
    T: plasma temperature (in K)
    r_c: corona radius (in gravitational radii)
    tau_T: Compton opacity
    eps_B: magnetization defined as U_mag/U_thermal,e
    p: spectral index of the electron energy distribution (must be > 2)
    delta: U_NT,e/U_th,e < 1
    '''
    M = M_BH * M_sun                     # BH mass
    R_g = 1.5e5 * (M/M_sun)              # Gravitational radius
    R_c = r_c * R_g                      # Coronal radius
    n_th0 = calc_nth0(M_BH, r_c, tau_T)  # Number density of thermal electrons
    theta_e = (k_B * T) / mec2           # Normalized temperature
    t_dyn = t_ff(r_c, M/M_sun)           # Characteristic dynamical timescale

    B = calc_B(M_BH=M_BH, r_c=r_c, tau_T=tau_T, T=T, eps_B=eps_B)   # Magnetic field in G

    Lnu = Lnu_fun(nu=nu, ne=n_th0, R=R_c, Theta=theta_e, B=B, t=t_dyn, delta=delta, p=p)
    return Lnu


def S_nu(nu_obs, z=0.01, M_BH=1.0, T=1e9, r_c=80, tau_T=1.0, eps_B=0.05,
         p=2.5, delta=0.1, D_L=None):
    '''Calculate the corona synchrotron flux at the frequencies "nu_obs".
    Output: fluxes in mJy.

    Parameters:
    -----------
    nu_obs: np.array with frequencies (in the observer reference frame)
    M_BH: BH mass in units of 10^8 M_sun
    T: plasma temperature (in K)
    r_c: corona radius (in gravitational radii)
    z: redshift (used to calculate the distance)
    tau_T: Compton opacity
    eps_B: magnetization defined as U_mag/U_thermal,e
    p: spectral index of the electron energy distribution (must be > 2)
    delta: U_NT,e/U_th,e < 1
    '''

    luminosity_distance = D_L if D_L is not None else Planck18.luminosity_distance(z)
    # if numeric D_L (Mpc), convert directly; else use .value from the astropy Quantity
    # if numeric D_L (Mpc), convert directly; else use .value from the astropy Quantity
    if hasattr(luminosity_distance, 'value'):
        d = luminosity_distance.value * 1e6 * pc
    else:
        d = float(luminosity_distance) * 1e6 * pc
    dil = 4 * pi * d**2
    nu = nu_obs * (1+z)
    # Make sure nu is always an array
    if np.isscalar(nu):
        nu = np.array([nu])

    Lnu = calculate_Lnu(nu=nu, M_BH=M_BH, T=T, r_c=r_c, tau_T=tau_T, eps_B=eps_B, p=p, delta=delta)
    S_nu = [L/dil/mJy for L in Lnu]
    return S_nu


def calc_nth0(M_BH, r_c, tau_T):
    M = M_BH * M_sun               # BH mass
    R_g = 1.5e5 * (M/M_sun)        # Gravitational radius
    R_c = r_c * R_g                # Coronal radius
    n_th0 = tau_T/(sigma_T * R_c)  # Number density of thermal electrons
    return n_th0


def calc_B(M_BH=1.0, r_c=80, tau_T=1.0, T=1e9, eps_B=0.05):
    n_th0 = calc_nth0(M_BH, r_c, tau_T)
    theta_e = (k_B * T) / mec2                      # Normalized temperature
    U_th = a_fun(theta_e) * n_th0 * theta_e * mec2  # Eq. 1 from Margalit+2021
    U_mag = eps_B * U_th
    B = np.sqrt(8 * pi * U_mag)                     # Magnetic field in G
    return B


def calc_Pth_factor(r_c=80, T=1e9):
    '''factor = P_th / U_th = (1 + T_i/T_e)/a(theta)'''
    theta_e = (k_B * T) / mec2   # Normalized temperature
    T_i = 9.1e10 * (20 / r_c)    # Ion temperature (Eq. 1 in Inoue+2024)
    return (1 + T_i / T) / a_fun(theta_e)


def calc_gmax_cool(eta_g, r_c, B):
    '''Calculate the maximum Lorentz factor of the electrons given by t_acc = t_syn.
    The formulae are taken from Inoue+2019 (https://iopscience.iop.org/article/10.3847/1538-4357/ab2715/pdf)
    eta_g: gyrofactor (mean free path in units of R_g)
    r_c: coronal radius in units of R_s
    B: magnetic field intensity'''
    # t_sy = 7.7e4 * (B / 10 G)**-2 * (gamma/100)**-1 s
    # t_acc = 7.6e-3 * (eta_g / 100) * (r_c/40) * (B / 10 G)**-1 * (gamma/100)**-1 s ! * (eta_acc/10)
    factor = (7.7e4/7.6e-3) * (100/eta_g) * (40/r_c) * (10/B)
    gmax_cool = 100 * np.sqrt(factor)
    return gmax_cool


def calc_gmax_conf(R, B):
    '''Calculate the maximum Lorentz factor of the electrons given by confinement (Hillas criterium).
    The formulae are taken from Inoue+2019 (https://iopscience.iop.org/article/10.3847/1538-4357/ab2715/pdf)
    R: accelerator size
    B: magnetic field intensity'''
    gmax_conf = q_e * B * R / mec2
    return gmax_conf