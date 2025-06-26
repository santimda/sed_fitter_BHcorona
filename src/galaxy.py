# galaxy.py
import os
import json
import numpy as np
from scipy.stats import gmean
import scipy.stats as stats
from bilby.core.utils import infer_parameters_from_function
from astropy.cosmology import Planck18

import matplotlib.pyplot as plt
import matplotlib
from plot_utils import nice_fonts
from plot_utils import plot_components

from constants import *
from utils import Omega_beam, get_f_sys, L100_to_LX, L100_to_LX2, L230_to_LX, L230_to_LX2, F100_to_FX, F230_to_FX
import corona_sed
from diffuse_emission import RJ_dust, pl_emission, pl_emission_abs, pl_emission_cl_abs


class Galaxy:
    def __init__(self, galaxy_name, use_kT_par=True, use_E_cond=True, eta_mag=0.75, xi_ep=10.0):
        self.name = galaxy_name
        self.use_kT_par = use_kT_par # Use Tc(tau) parametrization
        self.use_E_cond = use_E_cond # Use log_eps_B(log_delta) according to an energy condition
        self.eta_mag = eta_mag       # Value of U_mag/U_NT in the energy condition
        self.xi_ep = xi_ep           # Value of proton-to-electron ration in the energy condition
        self.seed_parameters = self.load_parameters()
        if self.use_kT_par:
            self.seed_parameters['kT'] = self.kT_par(
                tau_T=self.seed_parameters.get("tau_T"))
        if self.use_E_cond:
            self.seed_parameters['log_eps_B'] = self.E_cond(log_delta=self.seed_parameters.get(
                "log_delta"), eta_mag=self.eta_mag, xi_ep=self.xi_ep)

        self.seed_parameters['D_L'] = Planck18.luminosity_distance(
            self.seed_parameters.get("z"))

        # Initialize attributes based on the loaded parameters
        self.update_parameters(self.seed_parameters)

        # Load the observational data
        (self.data_frequencies,
            self.data_fluxes,
            self.data_fluxes_error,
            self.data_beams,
            self.data_type,
            self.data_frequencies_UL,
            self.data_fluxes_UL,
            self.data_fluxes_UL_error, 
            self.data_beams_UL,
            self.data_frequencies_LL,
            self.data_fluxes_LL,
            self.data_fluxes_LL_error,
            self.data_beams_LL
         ) = self.load_flux_data()

    def update_parameters(self, parameters):
        ''' Update the galaxy parameters'''
        parameter_names = ["z", "log_M", "r_c", "log_delta", "tau_T", "kT", "log_eps_B",
                           "p", "magnification", "alpha_sy", "beta_RJ", "tau1_freq",
                           "sy_scale", "ff_scale", "RJ_scale", "D_L", "nu_tau1_cl", "f_cov", "nu_tau1_diff"]

        for param_name in parameter_names:
            # Check if the parameter is in parameters dictionary before updating
            if param_name in parameters:
                setattr(self, param_name, parameters[param_name])

        # Special handling for kT and log_eps_B
        if self.use_kT_par:
            self.kT = self.kT_par(tau_T=self.tau_T)

        if self.use_E_cond:
            self.log_eps_B = self.E_cond(
                log_delta=self.log_delta, eta_mag=self.eta_mag, xi_ep=self.xi_ep)

    def load_parameters(self):
        ''' Load parameters from the parameters file'''
        parameter_file = f"../parameters/{self.name}.par"

        if not os.path.isfile(parameter_file):
            raise FileNotFoundError(
                f"Parameter file {parameter_file} not found.")

        with open(parameter_file, 'r') as file:
            parameters = json.load(file)

        return parameters

    def load_flux_data(self):
        ''' Load observational data information from the fluxes file.
        The errors are added systematic error uncertainties (get_f_sys function)'''
        flux_file = f"../fluxes/{self.name}_flux.dat"
        if not os.path.isfile(flux_file):
            raise FileNotFoundError(f"Flux data file {flux_file} not found.")

        data_frequencies, data_fluxes, data_fluxes_error, data_beams, data_type = [], [], [], [], []
        # Upper limits are indicated as data_type==1
        data_frequencies_UL, data_fluxes_UL, data_fluxes_UL_error, data_beams_UL = [], [], [], []
        # Lower limits are indicated as data_type==4
        data_frequencies_LL, data_fluxes_LL, data_fluxes_LL_error, data_beams_LL = [], [], [], []
        with open(flux_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # Skip empty lines and lines starting with "#"
                values = line.split()
                # The beam sizes are in arcsec x arcsec, and the beam size will be in stereoradian
                data_theta1, data_theta2 = float(
                    values[3])*arcsec, float(values[4])*arcsec
                sys_error_frac = get_f_sys(float(values[0]), data_theta1, data_theta2)
                flux_err = np.sqrt(float(values[2])**2 + (sys_error_frac*float(values[1]))**2)
                if int(values[5]) == 1:  # Check for data_type == 1 (ULs)
                    data_frequencies_UL.append(float(values[0]))
                    data_fluxes_UL.append(float(values[1]))
                    data_fluxes_UL_error.append(flux_err)
                    data_beams_UL.append(Omega_beam(data_theta1, data_theta2))
                elif int(values[5]) == 4:  # Check for data_type == 4 (LLs)
                    data_frequencies_LL.append(float(values[0]))
                    data_fluxes_LL.append(float(values[1]))
                    data_fluxes_LL_error.append(flux_err)
                    data_beams_LL.append(Omega_beam(data_theta1, data_theta2))
                else:
                    data_frequencies.append(float(values[0]))
                    data_fluxes.append(float(values[1]))
                    data_fluxes_error.append(flux_err)
                    data_type.append(int(values[5]))
                    data_beams.append(Omega_beam(data_theta1, data_theta2))

        return (
            np.array(data_frequencies),
            np.array(data_fluxes),
            np.array(data_fluxes_error),
            np.array(data_beams),
            np.array(data_type),
            np.array(data_frequencies_UL),
            np.array(data_fluxes_UL),
            np.array(data_fluxes_UL_error),
            np.array(data_beams_UL),
            np.array(data_frequencies_LL),
            np.array(data_fluxes_LL),
            np.array(data_fluxes_LL_error),
            np.array(data_beams_LL)
        )

    def plot_fluxes(self, xmin=None, xmax=None, ymin=None, ymax=None, title=None, show_beam_sizes=True, loglog_axes=True):
        '''
        Plot the observational data.

        Parameters:
            xmin (float): Minimum x-axis limit (default: None).
            xmax (float): Maximum x-axis limit (default: None).
            ymin (float): Minimum y-axis limit (default: None).
            ymax (float): Maximum y-axis limit (default: None).
            title (str): Plot title (default: None).
            show_beam_sizes (bool): Whether to show the beam sizes by scaling the poing sizes (default: True).
            loglog_axes (bool): Whether to use log-log axes (default: True).
        '''
        if title is None:
            title = self.name

        if xmin is None:
            xmin = 0.5 * np.min(np.concatenate(
                [self.data_frequencies, self.data_frequencies_UL, self.data_frequencies_LL]))

        if xmax is None:
            xmax = 2.0 * np.max(np.concatenate(
                [self.data_frequencies, self.data_frequencies_UL, self.data_frequencies_LL]))

        if ymin is None:
            ymin = 0.5 * \
                np.min(np.concatenate(
                    [self.data_fluxes, self.data_fluxes_UL, self.data_fluxes_LL]))

        if ymax is None:
            ymax = 2.0 * \
                np.max(np.concatenate(
                    [self.data_fluxes, self.data_fluxes_UL, self.data_fluxes_LL]))

        # Apply the nice_fonts settings
        matplotlib.rcParams.update(nice_fonts)

        fig, ax = plt.subplots(figsize=(8, 6))

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        if loglog_axes:
            ax.loglog()
        ax.tick_params(direction='in', which='both')

        # Plot the normal data points
        mask_type_0 = (self.data_type == 0)
        plt.errorbar(self.data_frequencies, self.data_fluxes,
                     yerr=self.data_fluxes_error, fmt='o', label="Data", markersize=3)

        # Plot upper limits with red arrows
        plt.errorbar(self.data_frequencies_UL, self.data_fluxes_UL, yerr=0.15*self.data_fluxes_UL,
                     uplims=True, fmt='o', color='C3', alpha=0.9, markersize=2, label='_nolegend_')

        # Plot lower limits with blue arrows
        plt.errorbar(self.data_frequencies_LL, self.data_fluxes_LL, yerr=0.15*self.data_fluxes_LL,
                     lolims=True, fmt='o', color='blue', alpha=0.9, markersize=2, label='_nolegend_')

        # Check if there are any fake points (data_type == 2) and plot them in a different colour
        if any(self.data_type == 2):
            mask_type_2 = (self.data_type == 2)
            plt.errorbar(self.data_frequencies[mask_type_2], self.data_fluxes[mask_type_2],
                         yerr=self.data_fluxes_error[mask_type_2], fmt='o', label="Fake data", color='C2', markersize=3)

        # Show the beam sizes scaling the point sizes
        if show_beam_sizes:
            average_beam = gmean(self.data_beams)
            scale_factor = 80.0
            point_sizes = scale_factor * \
                np.sqrt(self.data_beams / average_beam)
            plt.scatter(self.data_frequencies[mask_type_0], self.data_fluxes[mask_type_0],
                        s=point_sizes[mask_type_0], edgecolor='C1', facecolor='none', marker='o', alpha=1, label="Beam size")
            if any(self.data_type == 2):
                plt.scatter(self.data_frequencies[mask_type_2], self.data_fluxes[mask_type_2],
                            s=point_sizes[mask_type_2], edgecolor='C1', facecolor='none', marker='o', alpha=1, label="")

        ax.legend(loc="best", ncol=1, columnspacing=0.8)
        plt.xlabel(r"$\nu_{\rm obs}$ [GHz]")
        plt.ylabel(r"$S(\nu)$ [mJy]")
        plt.title(title)
        plt.grid(True)

        plt.show()

    def calculate_model(self, nu, z=None, magnification=None,
                        log_M=None, r_c=None, tau_T=None, log_delta=None, kT=None, log_eps_B=None, p=None,
                        alpha_sy=None, sy_scale=None, ff_scale=None,
                        beta_RJ=None, tau1_freq=None, RJ_scale=None, nu_tau1_cl=None, f_cov=None, nu_tau1_diff=None):
        ''' Calculate the SED for each individual component and the total emission. 
        It returns the total SED in mJy (used for fitting broadband data)'''

        # Set dynamic defaults
        log_M = log_M if log_M is not None else self.log_M
        r_c = r_c if r_c is not None else self.r_c
        tau_T = tau_T if tau_T is not None else self.tau_T
        log_delta = log_delta if log_delta is not None else self.log_delta
        kT = kT if kT is not None else (self.kT_par(
            tau_T=tau_T) if self.use_kT_par else self.kT)
        T = (kT / k_B) * keV  # Convert the temperature from kT[keV] to K
        log_eps_B = log_eps_B if log_eps_B is not None else (self.E_cond(
            log_delta=log_delta, eta_mag=self.eta_mag, xi_ep=self.xi_ep) if self.use_E_cond else self.log_eps_B)
        p = p if p is not None else self.p
        alpha_sy = alpha_sy if alpha_sy is not None else self.alpha_sy
        sy_scale = sy_scale if sy_scale is not None else self.sy_scale
        ff_scale = ff_scale if ff_scale is not None else self.ff_scale
        beta_RJ = beta_RJ if beta_RJ is not None else self.beta_RJ
        tau1_freq = tau1_freq if tau1_freq is not None else self.tau1_freq
        RJ_scale = RJ_scale if RJ_scale is not None else self.RJ_scale
        nu_tau1_cl = nu_tau1_cl if nu_tau1_cl is not None else self.nu_tau1_cl
        f_cov = f_cov if f_cov is not None else self.f_cov
        nu_tau1_diff = nu_tau1_diff if nu_tau1_diff is not None else self.nu_tau1_diff

        # Calculate each SED component and then the total SED as the sum
        self.S_cor = np.array(corona_sed.S_nu(nu, z=self.z, r_c=r_c, tau_T=tau_T, M_BH=(
            10**log_M), delta=(10**log_delta), T=T, eps_B=(10**log_eps_B), p=p, D_L=self.D_L)) * self.magnification
        self.S_dust = RJ_dust(nu, lg_scaling_rj=RJ_scale, nu_tau1_rest=tau1_freq,
                              beta=beta_RJ, z=self.z) * self.magnification
        self.S_ff = pl_emission_abs(nu, lg_scaling=ff_scale, alpha=-0.1, z=self.z, 
                                        nu_tau1_cl=nu_tau1_cl, nu_tau1_diff=nu_tau1_diff) * self.magnification
        self.S_sync = pl_emission_cl_abs(nu, lg_scaling=sy_scale, alpha=alpha_sy, z=self.z, 
                                            nu_tau1_cl=nu_tau1_cl, f_cov=f_cov, nu_tau1_diff=nu_tau1_diff) * self.magnification
        self.S_tot = self.S_cor + self.S_dust + self.S_ff + self.S_sync

        return self.S_tot

    def plot_model(self, nu):
        ''' Quick plot of the model'''
        title = self.name
        xmin, xmax = min(nu)/1e9, max(nu)/1e9
        ymin = 0.1 * \
            min(np.concatenate(
                [self.data_fluxes, self.data_fluxes_UL, self.S_tot]))
        ymax = 2.0 * \
            max(np.concatenate(
                [self.data_fluxes, self.data_fluxes_UL, self.S_tot]))

        # Apply the nice_fonts settings
        matplotlib.rcParams.update(nice_fonts)

        fig, ax = plt.subplots(figsize=(8, 6))

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        ax.loglog()
        ax.tick_params(direction='in', which='both')

        mask_type_0 = (self.data_type == 0)
        plt.errorbar(self.data_frequencies[mask_type_0], self.data_fluxes[mask_type_0],
                     yerr=self.data_fluxes_error[mask_type_0], label="Data", color='black', fmt='o')

        # Plot ULs and LLs with arrows in the same color 
        plt.errorbar(self.data_frequencies_UL, self.data_fluxes_UL, yerr=0.15*self.data_fluxes_UL,
                     uplims=True, fmt='o', color='black', alpha=0.9, markersize=2, label='_nolegend_')

        plt.errorbar(self.data_frequencies_LL, self.data_fluxes_LL, yerr=0.15*self.data_fluxes_LL,
                     lolims=True, fmt='o', color='black', alpha=0.9, markersize=2, label='_nolegend_')

        # Check if there are any fake points (data_type == 2)
        if any(self.data_type == 2):
            mask_type_2 = (self.data_type == 2)
            plt.errorbar(self.data_frequencies[mask_type_2], self.data_fluxes[mask_type_2],
                         yerr=self.data_fluxes_error[mask_type_2], label="Fake data", color='C2', fmt='o')

        plot_components(ax, nu,
                        [self.S_tot, self.S_cor, self.S_dust,
                            self.S_ff, self.S_sync],
                        show_legend=True, label_prefix='', alpha=1)

        ax.legend(loc="best", ncol=2, columnspacing=0.8)
        plt.xlabel(r"$\nu_{\rm obs}$ [GHz]")
        plt.ylabel(r"$S(\nu)$ [mJy]")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        plt.show()

    def get_fit_parameters(self, fit_parameters_raw):
        ''' Get an input list of parameters to fit. Check that the parameters are actually in the model
        parameters (if not remove them from the list and print a warning). Make a list containing the 
        fit_parameters in the correct order as given in the calculate_model function'''
        model_parameters = infer_parameters_from_function(self.calculate_model)

        invalid_parameters = set(fit_parameters_raw) - set(model_parameters)
        for param in invalid_parameters:
            print(
                f'\n WARNING: {param} is not a valid parameter! Removing it from the fit_parameter list \n')

        fit_parameters = []
        for param in model_parameters:
            if param in fit_parameters_raw:
                fit_parameters.append(param)

        return fit_parameters

    def set_prior_bounds(self, delta_z=0.5, delta_magnification=0.5,
                         delta_log_M=1.0, delta_r_c=None, delta_tau_T=3.0,
                         delta_log_delta=2.0, delta_kT=1.0, delta_log_eps_B=2.0, delta_p=1.0,
                         delta_alpha_sy=0.8, delta_sy_scale=1.5, delta_ff_scale=1.5,
                         delta_beta_RJ=0.3, delta_tau1_freq=3e2, delta_RJ_scale=1.5):
        ''' Set boundaries for the priors. The parameters "delta" give the explored range for each model parameter'''

        if delta_r_c is None:
            r_c_min, r_c_max = 10.0, 900.0
        else:
            r_c_min = max(self.seed_parameters['r_c'] - delta_r_c, 5.0)
            r_c_max = self.seed_parameters['r_c'] + delta_r_c 

        prior_bounds = {
            "z": (self.seed_parameters['z'] - delta_z, self.seed_parameters['z'] + delta_z),
            "magnification": (self.seed_parameters['magnification'] - delta_magnification, self.seed_parameters['magnification'] + delta_magnification),
            "log_M": (self.seed_parameters['log_M'] - delta_log_M, self.seed_parameters['log_M'] + delta_log_M),
            "r_c": (r_c_min, r_c_max),
            "tau_T": (0.01, 10.0),
            "log_delta": (self.seed_parameters['log_delta']-delta_log_delta, min(self.seed_parameters['log_delta']+delta_log_delta, 0.0)),
            "kT": (10.0, 210.0),
            "log_eps_B": (self.seed_parameters['log_eps_B']-delta_log_eps_B, min(self.seed_parameters['log_eps_B']+delta_log_eps_B, 2.0)),
            "p": (max(self.seed_parameters['p'] - delta_p, 2.1), self.seed_parameters['p'] + delta_p),
            "alpha_sy": (self.seed_parameters['alpha_sy'] - delta_alpha_sy, min(self.seed_parameters['alpha_sy'] + delta_alpha_sy, -0.45)),
            "sy_scale": (self.seed_parameters['sy_scale'] - delta_sy_scale, self.seed_parameters['sy_scale'] + delta_sy_scale),
            "ff_scale": (self.seed_parameters['ff_scale'] - delta_ff_scale, self.seed_parameters['ff_scale'] + delta_ff_scale),
            "beta_RJ": (1, 2),
            "tau1_freq": (100.0, 1200.0),
            "RJ_scale": (self.seed_parameters['RJ_scale'] - delta_RJ_scale, self.seed_parameters['RJ_scale'] + delta_RJ_scale),
            "nu_tau1_cl": (0, 30),
            "f_cov": (0, 1),
            "nu_tau1_diff": (0, 5),
        }
        return prior_bounds

    def seed_values(self):
        '''return the seed values in the parameter file'''
        log_M = self.seed_parameters.get("log_M")
        tau_T = self.seed_parameters.get("tau_T")
        tau1_freq = self.seed_parameters.get("tau1_freq")
        sy_scale = self.seed_parameters.get("sy_scale")
        ff_scale = self.seed_parameters.get("ff_scale")
        RJ_scale = self.seed_parameters.get("RJ_scale")
        return np.array([log_M, tau_T, tau1_freq, sy_scale, ff_scale, RJ_scale])

    def filter_data_by_beam(self, Omega_beam_min=None, Omega_beam_max=None):
        '''Method to remove data points for the fit, or treat points as ULs/LLs if they have 
        too low/high resolution.
        Note: use as UL the (flux + flux_err) and as LL the (flux - flux_err) value in order 
        to be conservative in case of loosely constrained fluxes, which gain too much weight 
        if they become strict upper/lower limits.'''

        # Check if the optional arguments are provided
        if Omega_beam_min is not None:
            # Filter data for elements where data_beam < Omega_beam_min
            indices_to_rm = [i for i, beam in enumerate(
                self.data_beams) if beam < Omega_beam_min]

            # Update lower limit lists
            self.data_frequencies_LL = np.concatenate(
                [self.data_frequencies_LL, self.data_frequencies[indices_to_rm]])
            self.data_fluxes_LL = np.concatenate([self.data_fluxes_LL, 
                                                  self.data_fluxes[indices_to_rm] - self.data_fluxes_error[indices_to_rm]])
            self.data_fluxes_LL_error = np.concatenate(
                [self.data_fluxes_LL_error, self.data_fluxes_error[indices_to_rm]])

           # Remove elements from the original lists
            self.data_frequencies = np.delete(self.data_frequencies, indices_to_rm)
            self.data_fluxes = np.delete(self.data_fluxes, indices_to_rm)
            self.data_fluxes_error = np.delete(self.data_fluxes_error, indices_to_rm)
            self.data_type = np.delete(self.data_type, indices_to_rm)
            self.data_beams = np.delete(self.data_beams, indices_to_rm)

        if Omega_beam_max is not None:
            # Filter data for elements where data_beam > Omega_beam_max
            indices_to_rm = [i for i, beam in enumerate(
                self.data_beams) if beam > Omega_beam_max]

            # Update upper limit lists
            self.data_frequencies_UL = np.concatenate(
                [self.data_frequencies_UL, self.data_frequencies[indices_to_rm]])
            self.data_fluxes_UL = np.concatenate([self.data_fluxes_UL,
                                                  self.data_fluxes[indices_to_rm] + self.data_fluxes_error[indices_to_rm]])
            self.data_fluxes_UL_error = np.concatenate(
                [self.data_fluxes_UL_error, self.data_fluxes_error[indices_to_rm]])

            # Remove elements from the original lists
            self.data_frequencies = np.delete(self.data_frequencies, indices_to_rm)
            self.data_fluxes = np.delete(self.data_fluxes, indices_to_rm)
            self.data_fluxes_error = np.delete(self.data_fluxes_error, indices_to_rm)
            self.data_type = np.delete(self.data_type, indices_to_rm)
            self.data_beams = np.delete(self.data_beams, indices_to_rm)

    def kT_par(self, tau_T):
        '''Use the value of Tc from the observed correlation between tau_T and T_c for a spherical geometry
        log10(kT) = a * log10(tau) + b, with (a, b) the coefficients from Tortosa+2018'''
        a, b = -0.7, 1.8
        log10_tau = np.log10(tau_T)
        log10_kT = a * log10_tau + b
        kT = np.power(10, log10_kT)
        return kT

    def E_cond(self, log_delta, eta_mag=1.0, xi_ep=40.0):
        '''Use the energy condition U_B = eta_mag * U_NT (eps_B = eta_mag * delta*(1+xi_ep) ) to derive eps_B from delta. 
        Note: eta_mag = 0.75 corresponds to the minimum energy condition
                xi_ep ~ 40 is the expected proton to electron ratio for shocks'''
        delta = np.power(10, log_delta)
        eps_B = eta_mag * delta * (1 + xi_ep)
        log_eps_B = np.log10(eps_B)
        return log_eps_B

    def B_G(self):
        '''Retrieve B[G]'''
        T = (self.kT / k_B) * keV  # Convert the temperature from kT[keV] to K
        return corona_sed.calc_B(M_BH=10**self.log_M, r_c=self.r_c, tau_T=self.tau_T, T=T, eps_B=(10**self.log_eps_B))

    def sigma_mag(self):
        '''Retrieve the magnetization parameter ( sigma_mag = B^2/(4pi n m_e c^2) )'''
        B = self.B_G()
        n_0 = corona_sed.calc_nth0(
            M_BH=10**self.log_M, r_c=self.r_c, tau_T=self.tau_T)
        return B**2 / (4. * pi * n_0 * mec2)
    
    def beta_mag(self):
        '''Retrieve the plasma beta parameter ( beta_mag = P_th/P_B = (P_i + P_e)/P_B )'''
        T = (self.kT / k_B) * keV  # Convert the temperature from kT[keV] to K
        factor = corona_sed.calc_Pth_factor(r_c=self.r_c, T=T)
        eps_B = np.power(10, self.log_eps_B)
        return factor / eps_B

    def R_c_cm(self):
        '''Return R_c in cm'''
        return self.r_c * 1.5e5 * np.power(10, self.log_M) 

    def E_cor(self):
        '''Retrieve the total energy in the corona in different components'''
        vol = (4./3.) * pi * self.R_c_cm()**3             # volume in cm3
        U_B = self.B_G()**2 / (8. * pi)                   # magnetic energy density
        E_B = U_B * vol
        eps_B = np.power(10, self.log_eps_B)
        delta = np.power(10, self.log_delta)
        E_th = E_B / eps_B
        E_nt = E_th * delta
        return E_th, E_nt, E_B

    def generate_all_y_samples(self, nu, accepted_samples, model_parameters, plot_samples=False):
        '''Generate samples for plotting confidence intervals'''
        all_S_samples = []
        all_Scor_samples = []
        all_Sdust_samples = []
        all_Sff_samples = []
        all_Ssy_samples = []
        for s in accepted_samples:
            s_values = [s[key] for key in model_parameters]
            self.calculate_model(nu, *s_values)
            all_S_samples.append(self.S_tot)
            all_Scor_samples.append(self.S_cor)
            all_Sdust_samples.append(self.S_dust)
            all_Sff_samples.append(self.S_ff)
            all_Ssy_samples.append(self.S_sync)
            # if plot_samples:
            #    plot_components(ax, nu, [self.S_tot, self.S_cor, self.S_dust, self.S_ff, self.S_sync],
            #                    show_legend=False, linestyle='--', label_prefix='mcmc', alpha=0.08)

        return np.array(all_S_samples), np.array(all_Scor_samples), np.array(all_Sdust_samples), np.array(all_Sff_samples), np.array(all_Ssy_samples)

    def calc_X(self, rel="R23", lum=True, verbose=True, corona_only=False, model_parameters=None):
        '''Calculate the intrinsic X-ray emission from the correlation between mm and X-ray emission.
        If lum=True it returns L_X(14-150 keV) in erg/s, else F_X(14-150 keV) in erg/s/cm2
        If verbose=True it prints a message with the emission at 2-10 keV and at 14-150 keV
        If corona_only=True it uses only the flux from the corona to estimate L_X, else the total flux

        Accepted relations:
        "R23": uses Ricci+ 2023 correlation between X-ray luminosity and 100 GHz luminosity (default)
        "K22": uses Kawamuro+ 2022 correlation between X-ray luminosity and 230 GHz luminosity
        '''
        if rel not in ["R23", "K22"]:
            print(f"Relation '{rel}' is not valid. Setting relation to Ricci+23 (R23)")
            rel = "R23"

        # Convert frequencies to the observed frame and calculate the SED
        nu = 100e9/(1 + self.z) if rel == "R23" else 230e9/(1 + self.z)     # in Hz
        self.calculate_model(nu, *model_parameters)

        # Use the whole flux or only the one coming from the corona to estimate L_X
        Fnu = self.S_cor[0] if corona_only else self.S_tot[0] 
        nuFnu = nu * Fnu * mJy               # nu*F_nu in erg/s/cm2

        # Convert flux to luminosity
        if lum:
            d = self.D_L.value * 1e6 * pc    # distance in cm
            dil = 4 * pi * d**2
            nuLnu = nuFnu * dil              # nu*L_nu in erg/s
            if rel == "R23" and lum:         
                X = L100_to_LX(nuLnu)      
                X2 = L100_to_LX2(nuLnu)
            else: 
                X = L230_to_LX(nuLnu)
                X2 = L230_to_LX2(nuLnu)
        else: 
            if rel == "R23": 
                X = F100_to_LX(nuFnu)
            else: 
                X = F230_to_LX(nuFnu)

        if verbose:
            if lum:
                if rel == "R23":
                    print(f"L_(100 GHz) = {nuLnu:.2e} erg/s")
                else: 
                    print(f"L_(230 GHz) = {nuLnu:.2e} erg/s")
                print(f"L_(2-10 keV) = {X2:.2e} erg/s")
                print(f"L_(14-150 keV) = {X:.2e} erg/s")
            else:
                print(f"F_(14-150 keV) = {X:.2e} erg/s/cm2")

        return X 


    def calculate_L_cor(self, nu):
        T = (self.kT / k_B) * keV  # Convert the temperature from kT[keV] to K
        self.L_cor = corona_sed.calculate_Lnu(
            nu=nu,
            M_BH=10**self.log_M,
            T=T,
            r_c=self.r_c,
            tau_T=self.tau_T,
            eps_B=10**self.log_eps_B,
            p=self.p,
            delta=10**self.log_delta
        )

    def calculate_L_bol_cor(self, nu=np.geomspace(1e9, 1e13, 200)):
        '''Calculate the bolometric luminosity of the corona (in erg/s)'''
        self.calculate_L_cor(nu)
        self.L_bol_cor = np.trapz(self.L_cor, nu)

    def calculate_L_peak_cor(self, nu=np.geomspace(1e9, 1e13, 200)):
        '''Calculate the peak luminosity of the corona, nu_p * L_p (in erg/s)'''
        self.calculate_L_cor(nu)
        nu_peak_index = np.argmax(self.L_cor)
        nu_peak = nu[nu_peak_index]
        L_peak = self.L_cor[nu_peak_index]
        self.L_peak_cor = nu_peak * L_peak    

    def calculate_Edd_ratio(self, L_X):
        '''Calculate the Eddington ratio (L_X / L_Edd) given the X-ray luminosity L_X.
        
        Parameters:
            L_X (float): The X-ray luminosity (in erg/s).
        
        Returns:
            float: The Eddington ratio (L_X / L_Edd).
        '''
        M_BH = 10 ** self.log_M
        L_Edd = 1.26e38 * M_BH  # erg/s
        lambda_edd = L_X / L_Edd
        return lambda_edd