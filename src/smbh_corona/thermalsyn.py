'''
Module taken from: https://github.com/bmargalit/thermal-synchrotron.
Some modifications were introduced to optimize the speed for MCMC fitting.

-------------

Calculates Synchrotron Emission from Thermal & Non-thermal Electrons

This module calculates synchrotron emission from a combined distribution of
thermal and non-thermal (power-law) electrons following the model presented in
Margalit & Quataert (2021; MQ21). The main function Lnu_shock() calculates the
emergent synchrotron spectral luminosity as a function of shock velocity,
radius, and upstream (ambient) density, accounting for synchrotron
self-absorption and fast cooling. Independently of the context of shock-powered
transients, synchrotron emission from a combined thermal + non-thermal electron
distribution can be calculated by specifying the electron density, emitting
region size, electron temperature, magnetic field, and system age, using the
function Lnu_fun().

Please cite Margalit & Quataert (2021) if used for scientific purposes:
https://ui.adsabs.harvard.edu/abs/2021arXiv211100012M/abstract

This file can be imported as a module and contains the following functions:
    * a_fun - eq. (1), MQ21
    * Theta_fun - eq. (2), MQ21
    * gamma_m_fun - eq. (6), MQ21
    * f_fun - eq. (5), MQ21
    * g_fun - eq. (8), MQ21
    * I_of_x - eq. (13), MQ21
    * C_j - eq. (15), MQ21
    * C_alpha - eq. (17), MQ21
    * low_freq_jpl_correction - low-frequency power-law emissivity
    * low_freq_apl_correction - low-frequency power-law absorption coefficient
    * jnu_pl - power-law emissivity; eqs. (14,19), MQ21
    * alphanu_pl - power-law absorption coefficient; eqs. (16,19), MQ21
    * jnu_th - thermal emissivity; eqs. (10,20), MQ21
    * alphanu_th - thermal absorption coefficient; eqs. (12,20), MQ21
    * tau_nu - total optical depth
    * nu_Theta - thermal frequency; eq. (11), MQ21
    * Lnu_fun - eq. (21), MQ21
    * Lnu_shock - main function; eq. (21), MQ21
    * spectral_index_fun - numerical calculation of spectral index
    * temporal_index_fun - numerical calculation of temporal index
    * find_xj - numerical calculation of frequency x_j
    * find_xalpha - numerical calculation of frequency x_alpha
'''

import numpy as np
from scipy import special
from scipy import optimize
from functools import lru_cache

# define physical constants (in cgs units):
q = 4.80320425e-10
c = 2.99792458e10
me = 9.1093837015e-28
mp = 1.673e-24
sigT = 6.652e-25

# Decorator for memoization (caching)


@lru_cache(maxsize=None)
def memoize(func):
    return func


@memoize
def a_fun(Theta):
    '''Utility function defined in eq. (1), MQ21

    This is an approximate fitting function from Gammie & Popham (1998)

    Parameters
    __________
    Theta : float
        The dimensionless electron temperature

    Returns
    _______
    a : float
        value of a(Theta)
    '''

    val = (6.0+15.0*Theta)/(4.0+5.0*Theta)
    return val


@memoize
def Theta_fun(beta, mu=0.62, mu_e=1.18, epsilon_T=1e0):
    '''Calculate electron temperature

    This calculates the post-shock dimensionless electron temperature following
    eqs. (2,3) of MQ21

    Parameters
    __________
    beta : float
        Shock velocity / c
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18
    epsilon_T : float, optional
        Electron thermalization efficiency (<=1), default is 1e0

    Returns
    _______
    Theta : float
        Dimensionless electron temperature = kb*Te/me*c**2
    '''

    # eq. (3) of MQ21:
    Theta0 = epsilon_T*(9.0*mu*mp/(32.0*mu_e*me))*beta**2
    # eq. (2) of MQ21:
    val = (5.0*Theta0-6.0+(25.0*Theta0**2+180.0*Theta0+36.0)**0.5)/30.0
    return val


@memoize
def gamma_m_fun(Theta):
    '''Calculate minimal Lorentz factor of non-thermal electrons; eq. (6), MQ21

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature

    Returns
    _______
    gamma_m : float
        minimum Lorentz factor of power-law electrons
    '''

    return 1e0+a_fun(Theta)*Theta


@memoize
def f_fun(Theta):
    '''Utility function f(Theta); eq. (5) of MQ21

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature

    Returns
    _______
    f : float
        Correction term to the thermal electron distribution that is relevant
        only in the non-relativistic regime (Theta ~< 1)
    '''

    return 2.0*Theta**2/special.kn(2, 1.0/Theta)


@memoize
def g_fun(Theta, p=3.0):
    '''Utility function g(Theta); eq. (8) of MQ21
    Important!: Fixed error according to Margalit+24
    (priv. comm., new repo: https://github.com/bmargalit/thermal-synchrotron-v2/blob/main/thermalsyn_v2.py)

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    p : float, optional
        Slope of power-law electron distribution, default is 3.0

    Returns
    _______
    g : float
        Correction term to the power-law electron distribution that is relevant
        only in the non-relativistic regime (Theta ~< 1)
    '''

    gamma_m = gamma_m_fun(Theta)
    # val = ( (p-1.0)*gamma_m / ( (p-1.0)*gamma_m - p+2.0 ) )*(gamma_m/(3.0*Theta))**(p-1.0) # slightly rewritten version, before correction
    val = ((p-1.0)*a_fun(Theta)*Theta/((p-1.0)*gamma_m - p+2.0)) * \
        (gamma_m/(3.0*Theta))**(p-1.0)  # v2.0
    return val


def I_of_x(x):
    '''Function I'(x) derived by Mahadevan et al. (1996)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)

    Returns
    _______
    I : float
        Spectral energy distribution function (eq. 13, MQ21)
    '''

    return 4.0505*x**(-1.0/6.0)*(1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5)) * np.exp(-1.8899*np.cbrt(x))


@memoize
def C_j(p):
    '''Prefactor to power-law synchrotron emissivity (eq. 15, MQ21)

    Parameters
    __________
    p : float
        Slope of power-law electron distribution

    Returns
    _______
    Cj : float
        Synchrotron constant
    '''

    return (special.gamma((p+5.0)/4.0)/special.gamma((p+7.0)/4.0))*special.gamma((3.0*p+19.0)/12.0)*special.gamma((3.0*p-1.0)/12.0)*((p-2.0)/(p+1.0))*3.0**((2.0*p-1.0)/2.0)*2.0**(-(7.0-p)/2.0)*np.pi**(-0.5)


@memoize
def C_alpha(p):
    '''Prefactor to power-law synchrotron absorption coefficient (eq. 17, MQ21)

    Parameters
    __________
    p : float
        Slope of power-law electron distribution

    Returns
    _______
    Calpha : float
        Synchrotron constant
    '''

    return (special.gamma((p+6.0)/4.0)/special.gamma((p+8.0)/4.0))*special.gamma((3.0*p+2.0)/12.0)*special.gamma((3.0*p+22.0)/12.0)*(p-2.0)*3.0**((2.0*p-5.0)/2.0)*2.0**(p/2.0)*np.pi**(3.0/2.0)


@memoize
def Cj_low_calc(p):
    return -np.pi**1.5*(p-2.0)/(2.0**(1.0/3.0)*3.0**(1.0/6.0)*(3.0*p-1.0)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0))


def low_freq_jpl_correction(x, Theta, p):
    '''Low-frequency correction to power-law emissivity

    This function returns a multiplicative frequency-dependent correction term
    that modifies the high-frequency power-law emissivity in the regime
    nu <~ nu_m (where nu_m is the synchrotron frequency corresponding to the
    minimum Lorentz factor of power-law electrons, gamma_m). We adopt an
    approximate expression that smoothly interpolates between the exact high-
    and low-frequency results.

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    Theta : float
        Dimensionless electron temperature
    p : float
        Slope of power-law electron distribution

    Returns
    _______
    val : float
        Correction term
    '''

    gamma_m = gamma_m_fun(Theta)
    # synchrotron constant in x<<x_m limit
    Cj_low = Cj_low_calc(p)
    # multiplicative correction term
    corr = (Cj_low/C_j(p))*(gamma_m/(3.0*Theta)
                            )**(-(3.0*p-1.0)/3.0)*x**((3.0*p-1.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    val = (1e0 + corr**(-s))**(-1.0/s)
    return val


@memoize
def calculate_Calpha_low(p):
    return -2.0**(8.0/3.0)*np.pi**(7.0/2.0)*(p+2.0)*(p-2.0)/(3.0**(19.0/6.0)*(3.0*p+2)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0))


def low_freq_apl_correction(x, Theta, p):
    '''Low-frequency correction to power-law absorption coefficient

    This function returns a multiplicative frequency-dependent correction term
    that modifies the high-frequency power-law absorption coeff in the regime
    nu <~ nu_m (where nu_m is the synchrotron frequency corresponding to the
    minimum Lorentz factor of power-law electrons, gamma_m). We adopt an
    approximate expression that smoothly interpolates between the exact high-
    and low-frequency results.

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    Theta : float
        Dimensionless electron temperature
    p : float
        Slope of power-law electron distribution

    Returns
    _______
    val : float
        Correction term
    '''

    gamma_m = gamma_m_fun(Theta)
    # synchrotron constant in x<<x_m limit
    Calpha_low = calculate_Calpha_low(p)
    # multiplicative correction term
    corr = (Calpha_low/C_alpha(p))*(gamma_m/(3.0*Theta)
                                    )**(-(3.0*p+2.0)/3.0)*x**((3.0*p+2.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    val = (1e0 + corr**(-s))**(-1.0/s)
    return val


def jnu_pl(x, n, B, Theta, delta=1e-1, p=3.0, z_cool=np.inf):
    '''Synchrotron emissivity of power-law electrons (eqs. 14,19; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron emissivity
    '''

    gamma_m = gamma_m_fun(Theta)
    val = C_j(p)*(q**3/(me*c**2))*delta*n*B*g_fun(Theta, p=p)*x**(-(p-1.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= low_freq_jpl_correction(x, Theta, p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum(1e0, (z_cool/z0))
    return val


def alphanu_pl(x, n, B, Theta, delta=1e-1, p=3.0, z_cool=np.inf):
    '''Synchrotron absorption coeff of power-law electrons (eqs. 16,19; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron absorption coefficient
    '''

    gamma_m = gamma_m_fun(Theta)
    val = C_alpha(p)*q*(delta*n/(Theta**5*B)) * \
        g_fun(Theta, p=p)*x**(-(p+4.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= low_freq_apl_correction(x, Theta, p)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum(1e0, (z_cool/z0))
    return val


def jnu_th(x, n, B, Theta, z_cool=np.inf):
    '''Synchrotron emissivity of thermal electrons (eqs. 10,20; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron emissivity
    '''

    val = (3.0**0.5/(8.0*np.pi))*(q**3/(me*c**2))*f_fun(Theta)*n*B*x*I_of_x(x)
    # fast-cooling correction:
    z0 = np.cbrt(2.0*x)
    val *= np.minimum(1e0, z_cool/z0)
    return val


def alphanu_th(x, n, B, Theta, z_cool=np.inf):
    '''Synchrotron absorption coeff of thermal electrons (eqs. 12,20; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron absorption coefficient
    '''

    val = (np.pi*3.0**(-3.0/2.0))*q*(n/(Theta**5*B))*f_fun(Theta)/x * I_of_x(x)
    # fast-cooling correction:
    z0 = np.cbrt(2.0*x)
    val *= np.minimum(1e0, z_cool/z0)
    return val


def tau_nu(x, n, R, B, Theta, delta=1e-1, p=3.0, z_cool=np.inf):
    '''Total (thermal+non-thermal) synchrotron optical depth

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    R : float
        Characteristic size of the emitting region (in cm)
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron absorption coefficient
    '''

    val = R*(alphanu_th(x, n, B, Theta, z_cool=z_cool) +
             alphanu_pl(x, n, B, Theta, delta=delta, p=p, z_cool=z_cool))
    return val


@memoize
def nu_Theta(Theta, B):
    '''Characteristic "thermal" synchrotron frequency (eq. 11, MQ21)

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    B : float
        Magnetic field strength (in G)

    Returns
    _______
    val : float
        Synchrotron frequency
    '''

    val = 3.0*Theta**2*q*B/(4.0*np.pi*me*c)
    return val


def Lnu_fun(nu, ne, R, Theta, B, t, delta=1e-1, p=3.0):
    '''Synchrotron specific luminosity

    Calculates the emergent synchrotron luminosity at frequency
    nu (including synchrotron self-absorption and synchrotron cooling effects)
    as a function of the physical parameters within the emitting region. This
    form is applicable also outside the scope of shock-powered transients.

    Parameters:
    -----------
    nu : float
        Frequency (in Hz)
    ne : float
        Electron number density in the emitting region (in cm^{-3})
    R : float
        Characteristic size of the emitting region (in cm)
    Theta : float
        Dimensionless electron temperature
    B : float
        Magnetic field strength (in G)
    t : float
        Dynamical time = R/v ~ source age (in s)
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0

    Returns:
    --------
    val : float
        Specific luminosity (in erg/s/Hz)
    '''

    # calculate the (normalized) cooling Lorentz factor (eq. 18, MQ21):
    z_cool = (6.0*np.pi*me*c/(sigT*B**2*t))/Theta
    # normalized frequency:
    x = nu/nu_Theta(Theta, B)
    # calculate total emissivity & optical depth:
    j = jnu_th(x, ne, B, Theta, z_cool=z_cool) + \
        jnu_pl(x, ne, B, Theta, delta=delta, p=p, z_cool=z_cool)
    tau = tau_nu(x, ne, R, B, Theta, delta=delta, p=p, z_cool=z_cool)
    # Calculate specific luminosity Lnu (eq. 21, MQ21):
    val = 4.0 * np.pi**2 * R**3 * j
    # Incorporate the transmission factor, which is (1-e^-tau)/tau ~ 1 for tau << 1:
    val *= np.where(tau < 1e-4, 1.0, (1e0 - np.exp(-tau)) / tau)
    return val


def Lnu_shock(nu, n, R, v, epsilon_T=1e0, epsilon_B=1e-1, delta=1e-1, p=3.0, mu=0.62, mu_e=1.18):
    '''Synchrotron specific luminosity as a function of shock parameters

    This function calculates the emergent synchrotron luminosity at frequency
    nu (including synchrotron self-absorption and synchrotron cooling effects)
    within the context of a shock-powered model: the post-shock magnetic field,
    electron temperature, and electron number-density are related to the shock
    velocity and upstream (ambient medium) density using epsilon_B & epsilon_T.

    Parameters
    __________
    nu : float
        Frequency (in Hz)
    n : float
        *Upstream* number density (in cm^{-3})
    R : float
        Characteristic size of the emitting region (in cm)
    v : float
        Shock velocity (in g/s)
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 1e0
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        Specific luminosity (in erg/s/Hz)
    '''

    # downstream (post-shock) electron number density:
    ne = 4.0*mu_e*n
    # downstream electron temperature:
    Theta = Theta_fun(v/c, mu=mu, epsilon_T=epsilon_T)
    # shock-amplified magnetic field (eq. 9; MQ21):
    B = (9.0*np.pi*epsilon_B*n*mu*mp)**0.5*v
    # mean dynamical time:
    t = R/v
    # calculate luminosity:
    val = Lnu_fun(nu, ne, R, Theta, B, t, delta=delta, p=p)
    return val


def spectral_index_fun(nu, n, R, v, epsilon_T=1e0, epsilon_B=1e-1, delta=1e-1, p=3.0, mu=0.62, mu_e=1.18):
    '''Spectral index at frequency nu

    Parameters
    __________
    nu : float
        Frequency (in Hz)
    n : float
        *Upstream* number density (in cm^{-3})
    R : float
        Characteristic size of the emitting region (in cm)
    v : float
        Shock velocity (in g/s)
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 1e0
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        Spectral index = dln(L_nu)/dln(nu)
    '''

    nu_vec = nu*np.array([0.99, 1.01])
    nu_ary, n_ary = np.meshgrid(nu_vec, n)
    nu_ary, R_ary = np.meshgrid(nu_vec, R)
    nu_ary, v_ary = np.meshgrid(nu_vec, v)
    Lnu_ary = Lnu_shock(nu_ary, n_ary, R_ary, v_ary, epsilon_T=epsilon_T,
                        epsilon_B=epsilon_B, delta=delta, p=p, mu=mu, mu_e=mu_e)
    val = np.log10(Lnu_ary[:, 1]/Lnu_ary[:, 0])/np.log10(nu_vec[1]/nu_vec[0])
    return val


def temporal_index_fun(t, nu, n, R, v, epsilon_T=1e0, epsilon_B=1e-1, delta=1e-1, p=3.0, mu=0.62, mu_e=1.18):
    '''Temporal index at frequency nu

    Parameters
    __________
    nu : float
        Frequency (in Hz)
    n : float
        *Upstream* number density (in cm^{-3})
    R : float
        Characteristic size of the emitting region (in cm)
    v : float
        Shock velocity (in g/s)
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 1e0
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        Temporal index = dln(L_nu)/dln(t)
    '''

    Lnu = Lnu_shock(nu, n, R, v, epsilon_T=epsilon_T,
                    epsilon_B=epsilon_B, delta=delta, p=p, mu=mu, mu_e=mu_e)
    val = np.zeros_like(Lnu)
    val[1:-1] = np.log10(Lnu[2:]/Lnu[:-2])/np.log10(t[2:]/t[:-2])
    val[0] = np.log10(Lnu[1]/Lnu[0])/np.log10(t[1]/t[0])
    val[-1] = np.log10(Lnu[-1]/Lnu[-2])/np.log10(t[-1]/t[-2])
    return val


def find_xj(Theta, delta=1e-1, p=3.0, z_cool=np.inf):
    '''Numerically solve for x_j, the power-law/thermal transition frequency

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    x_j : float
        The (dimensionless) frequency x_j
    '''

    # arbitrary values for ne and B---the result is independent of this choice:
    n = 1e0
    B = 1e0
    # root-finding:
    x_j = 10**optimize.fsolve(lambda x: np.log10(jnu_pl(10**x, n, B, Theta, delta=delta,
                              p=p, z_cool=z_cool)/jnu_th(10**x, n, B, Theta, z_cool=z_cool)), x0=2.0)[0]
    return x_j


def find_xalpha(Theta, delta=1e-1, p=3.0, z_cool=np.inf):
    '''Numerically solve for x_alpha

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    x_j : float
        The (dimensionless) frequency x_alpha
    '''

    # arbitrary values for ne and B---the result is independent of this choice:
    n = 1e0
    B = 1e0
    # root-finding:
    x_alpha = 10**optimize.fsolve(lambda x: np.log10(alphanu_pl(10**x, n, B, Theta, delta=delta,
                                  p=p, z_cool=z_cool)/alphanu_th(10**x, n, B, Theta, z_cool=z_cool)), x0=3.0)[0]
    return x_alpha
