#!/usr/bin/env python
# coding: utf-8

import numpy as np

from bilby.core.likelihood import Likelihood
from bilby.core.utils import infer_parameters_from_function

from constants import *


def Sobs_to_TB(S_nu, nu, Omega_b):
    '''Converts the fluxes S_nu at the frequencies nu to brightness temperature T_B. 
    Omega_b are the solid angles of the beam at each frequency. Similar to Snu_to_TB
    but without knowledge of the true size of the source.

    Input:
    [S_nu] = mJy 
    [nu] = Hz 
    [Omega_b] = stereoradian

    Output:
    [T_B] = K
    '''
    return c**2 * (S_nu*mJy) / (2 * k_B * np.power(nu, 2) * Omega_b)


def Omega_beam(theta1, theta2):
    '''Beam solid angle in terms of the theta_HPBW (theta1 x theta2)
    Use the equation here: https://science.nrao.edu/facilities/vla/proposing/TBconv'''
    return pi * theta1 * theta2 / (4 * np.log(2))


def reduced_chi_square(observed, model, errors, num_params):
    '''Function to calculate the reduced chi square. 
    num_params: number of fitted parameters'''
    residuals = (observed - model) / errors
    chi_square = np.sum(residuals**2)
    # degrees of freedom = number of data points - number of parameters
    dof = len(observed) - num_params
    reduced_chi_square = chi_square / dof
    return reduced_chi_square


def get_f_sys(nu, theta1, theta2):
    """
    Calculate the systematic error in fluxes (f_sys)
    We assume it is 5% for all observatories except for 
    ALMA in B6-8 (10%) and B9-10 (20%). ALMA is the only
    instrument with < 0.5" resolution in these frequency ranges
    (maybe NOEMA as well?).

    Parameters:
    - nu (float): frequency in GHz
    - theta1, theta2 (float): beam size in radians

    Returns:
    - float: The systematic error in fluxes (f_sys).
    """
    if theta1 < 0.5 * arcsec or theta2 < 0.5 * arcsec:
        if 212 <= nu <= 500:
            f_sys = 0.1
        elif 600 <= nu <= 900:
            f_sys = 0.2
        else:
            f_sys = 0.05
    else:
        f_sys = 0.05
    return f_sys

#=================================================================================================

def calculate_logX(logY, alpha, beta):
    """
    Function to calculate linear relations between quantities as defined in Table 2 in Kawamuro+22:
    logY = alpha * logX + beta
    
    Parameters:
    logY (float): The logarithm of Y value.
    alpha (float): The alpha parameter.
    beta (float): The beta parameter.
    
    Returns:
    float: The calculated logX value.
    """
    logX = (logY - beta) / alpha
    return logX

def L230_to_LX2(L230):
    """
    Calculate the X-ray luminosity in the 2-10 keV range using 
    the linear relation from Kawamuro+22 (Table 2) for RQ AGNs.
    
    Parameters:
    L230 (float): The value of nu*L_nu at nu=230 GHz (in erg/s)
    
    Returns:
    float: The calculated LX value in erg/s
    """
    alpha = 1.08  # \pm 0.06   
    beta = -7.41 # +2.25 -2.71 
    log_LX = calculate_logX( np.log10(L230), alpha, beta )
    return 10**log_LX

def L230_to_LX(L230):
    """
    Calculate the X-ray luminosity in the 14-150 keV range using 
    the linear relation from Kawamuro+22 (Table 2) for the full sample of AGNs.
    
    Parameters:
    L230 (float): The value of nu*L_nu at nu=230 GHz (in erg/s)
    
    Returns:
    float: The calculated LX value in erg/s
    """
    alpha = 1.19   
    beta = -12.78 # \pm 0.05 
    log_LX = calculate_logX( np.log10(L230), alpha, beta )
    return 10**log_LX

def F230_to_FX(F230):
    """
    Calculate the X-ray flux in the 14-150 keV range using 
    the linear relation from Kawamuro+22 (Table 2) for RQ AGNs.
    
    Parameters:
    F230 (float): The value of nu*F_nu at nu=230 GHz (in erg/s/cm2)
    
    Returns:
    float: The calculated FX value in erg/s/cm2
    """
    alpha = 1.17  # +0.17 -0.13   
    beta = -2.73 # +1.8 -1.3 
    log_FX = calculate_logX( np.log10(F230), alpha, beta )
    return 10**log_FX

def L100_to_LX2(L100):
    """
    Calculate the X-ray luminosity in the 2-10 keV range using 
    the linear relation from Ricci+23 (Eq. 3).
    
    Parameters:
    L100 (float): The value of nu*L_nu at nu=100 GHz (in erg/s)
    
    Returns:
    float: The calculated LX value in erg/s
    """
    alpha = 1.22 # \pm 0.02  
    beta = -13.9 # \pm 0.8 
    log_LX = calculate_logX( np.log10(L100), alpha, beta )
    return 10**log_LX

def L100_to_LX(L100):
    """
    Calculate the X-ray luminosity in the 14-150 keV range using 
    the linear relation from Ricci+23 (Eq. 1).
    
    Parameters:
    L100 (float): The value of nu*L_nu at nu=100 GHz (in erg/s)
    
    Returns:
    float: The calculated LX value in erg/s
    """
    alpha = 1.22 # \pm 0.02  
    beta = -14.4 # \pm 0.8 
    log_LX = calculate_logX( np.log10(L100), alpha, beta )
    return 10**log_LX

def F100_to_FX(F100):
    """
    Calculate the X-ray flux in the 14-150 keV range using 
    the linear relation from Ricci+23 (Eq. 2).
    
    Parameters:
    F100 (float): The value of nu*F_nu at nu=100 GHz (in erg/s/cm2)
    
    Returns:
    float: The calculated FX value in erg/s/cm2
    """
    alpha = 1.37 # \pm 0.04  
    beta = -1.3  # \pm 0.4
    log_FX = calculate_logX( np.log10(F100), alpha, beta )
    return 10**log_FX

def LX_to_Lbol(LX):
    """
    Convert the X-ray luminosity to the bolometric luminosity using 
    the conversion factor from Ricci+23 

    Parameters:
    LX (float): The value of L_(14-150 keV) in erg/s
    
    Returns:
    float: L_bol in erg/s
    """
    factor = 8.48  # in the 2-10 keV it is 20 (not implemented in the code)
    return factor * LX


def format_with_errors(value, bounds, decimals=2):
    """
    Formats a value with errors in compact (a±b)exponent or (a+b-c)exponent form.
    
    Args:
        value (float): Central value.
        bounds (float or tuple): 
            - If float: symmetric bounds (value ± bounds).
            - If tuple (lower_bound, upper_bound): asymmetric bounds.
        decimals (int): Decimal places (default: 2).
    
    Returns:
        str: Formatted string like "(1.23±0.45)e42" or "(1.23+0.45-0.67)e42".
    """
    # Calculate errors from bounds
    if isinstance(bounds, (float, int)):
        err_low = err_high = bounds  # Symmetric case
    elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        lower, upper = bounds
        err_low = value - lower  # Calculate lower error
        err_high = upper - value  # Calculate upper error
    else:
        raise ValueError("`bounds` must be a number (symmetric) or tuple (lower, upper).")

    # Get exponent and scale values
    exponent = int(f"{value:.{decimals}e}".split('e')[-1])
    scale = 10 ** exponent
    scaled_val = value / scale
    scaled_high = err_high / scale
    scaled_low = err_low / scale

    # Return formatted string
    if err_low == err_high:
        return f"({scaled_val:.{decimals}f}±{scaled_high:.{decimals}f})e{exponent}"
    else:
        return f"({scaled_val:.{decimals}f}+{scaled_high:.{decimals}f}-{scaled_low:.{decimals}f})e{exponent}"

#=================================================================================================


class Analytical1DLikelihood_withUL_andLL(Likelihood):
    """
    A general class for 1D analytical functions. The model
    parameters are inferred from the arguments of the function.
    An addition to manage upper limits (UL) and lower limits (LL) was introduced.

    Parameters
    ==========
    x, y: array_like
        The data to analyse
    func:
        The python function to fit to the data. Note, this must take the
        dependent variable as its first argument. The other arguments
        will require a prior and will be sampled over (unless a fixed
        value is given).
    x_ul, y_ul: array_like, optional
        The upper limit data. If provided, the exceed_ul method will check if
        self.func(x_ul, **self.model_parameters) > y_ul.
    x_ll, y_ll: array_like, optional
        The lower limit data. If provided, the below_ll method will check if
        self.func(x_ll, **self.model_parameters) < y_ll.
    """

    def __init__(self, x, y, func, x_ul=None, y_ul=None, x_ll=None, y_ll=None):
        parameters = infer_parameters_from_function(func)
        super(Analytical1DLikelihood_withUL_andLL, self).__init__(dict.fromkeys(parameters))
        self.x = x
        self.y = y
        self._func = func
        self._function_keys = list(self.parameters.keys())
        self.x_ul = x_ul
        self.y_ul = y_ul
        self.x_ll = x_ll
        self.y_ll = y_ll

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, func={self.func.__name__}, x_ul={self.x_ul}, y_ul={self.y_ul}, x_ll={self.x_ll}, y_ll={self.y_ll})"
        
    @property
    def func(self):
        """ Make func read-only """
        return self._func

    @property
    def model_parameters(self):
        """ This sets up the function only parameters (i.e. not sigma for the GaussianLikelihood) """
        return {key: self.parameters[key] for key in self.function_keys}

    @property
    def function_keys(self):
        """ Makes function_keys read_only """
        return self._function_keys

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)

    @property
    def x(self):
        """ The independent variable. Setter assures that single numbers will be converted to arrays internally """
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        self._x = x

    @property
    def y(self):
        """ The dependent variable. Setter assures that single numbers will be converted to arrays internally """
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self._y = y

    @property
    def residual(self):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.model_parameters)

    @property
    def exceed_ul(self):
        """Check if model predictions exceed upper limits"""
        if self.x_ul is not None and self.y_ul is not None:
            model_predictions_ul = self.func(self.x_ul, **self.model_parameters)
            return np.any(model_predictions_ul > self.y_ul)
        return False

    @property
    def below_ll(self):
        """Check if model predictions are below lower limits"""
        if self.x_ll is not None and self.y_ll is not None:
            model_predictions_ll = self.func(self.x_ll, **self.model_parameters)
            return np.any(model_predictions_ll < self.y_ll)
        return False


class GaussianLikelihood_withUL_andLL(Analytical1DLikelihood_withUL_andLL):
    def __init__(self, x, y, func, sigma=None, x_ul=None, y_ul=None, x_ll=None, y_ll=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of the function.

        Parameters
        ==========
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        x_ul, y_ul: array_like, optional
            The upper limit data. If provided, the exceed_ul method will check if
            self.func(x_ul, **self.model_parameters) > y_ul.
        x_ll, y_ll: array_like, optional
            The lower limit data. If provided, the below_ll method will check if
            self.func(x_ll, **self.model_parameters) < y_ll.
        """

        super(GaussianLikelihood_withUL_andLL, self).__init__(x=x, y=y, func=func, x_ul=x_ul, y_ul=y_ul, x_ll=x_ll, y_ll=y_ll)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        # Check if any model predictions exceed any upper limits or are below any lower limits
        if self.exceed_ul or self.below_ll:
            return -1e30 #-np.inf

        # Calculate log-likelihood for detected data points
        log_l = np.sum(- (self.residual / self.sigma)**2 / 2 - np.log(2 * np.pi * self.sigma**2) / 2)
    
        return log_l

    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, func={self.func.__name__}, sigma={self.sigma}, x_ul={self.x_ul}, y_ul={self.y_ul}, x_ll={self.x_ll}, y_ll={self.y_ll})"

    @property
    def sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like with length equal to x.')
