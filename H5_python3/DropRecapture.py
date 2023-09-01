from typing import Union, Tuple, List
from physics import *
from scipy.interpolate import interp1d
import numpy as np
from Iterations import Iterations
from scipy.optimize import curve_fit

itar = Union[float,ndarray]

tau = 2*pi


def dephase_SHO(xi,vi,o):
    rg = -vi/(o*xi)
    A = xi*sqrt(1+rg**2)
    phi = random.random(len(xi))*2*pi
    xf = A*cos(phi)
    vf = -A*o*sin(phi)
    return xf, vf


def SHO_ic(
        temperature: float,
        p: float,
        w0: float,
        alpha_0: float = alpha_0_938,
        rayleigh: float = None,
        wl: float = 938e-9,
        n: int = 1000,
        v0: List =None,
        scramble = False
):
    """
    Samples a Maxwell-Boltzmann distribution of initial conditions for a thermal sample of point particles in a (nearly) SHO potential
    corrected to contain anharmonicities in a red tweezer ODT.
    Args:
        temperature: temperature of atom (K)
        p: power of the ODT beam (W)
        w0: waist of the ODT beam (m)
        alpha_0: atom polarizability at ODT wavelength. Default is alpha_0 for Cs at 938nm. (Jm^2/V^2)
        rayleigh: rayleigh range of the ODT beam. Computed from w0 and wl if not provided. (m)
        m: mass of the trapped atom. Default is mass of cesium (kg)
        wl: wavelength of trapping light. Default is 938nm (m)
        n: number of times to repeat the MC simulation. Default if 1000
        v0: default [0,0,0]. If provided and non-zero the Maxwell-Boltzmann distribution for velocities will
            be centered around v0[i] instead of zero.

    Returns:
        [xi,yi,zi,vxi,vyi,vzi]
    """
    if rayleigh is None:
        rayleigh = zr(w0, wl)

    if v0 is None:
        v0 = [0, 0, 0]

    # print(w0,rayleigh)
    freq_t = trap_frequency_t(p, w0, alpha_0)
    freq_a = trap_frequency_a(p, w0, alpha_0, rayleigh=rayleigh)
    # print(freq_a/tau,freq_t/tau)

    std_t = sigma_x_boltz_SHO(temperature, freq_t)
    std_a = sigma_x_boltz_SHO(temperature, freq_a)
    std_v = sigma_v_boltz(temperature)
    # print(std_t,std_a,std_v)

    # establish initial conditions
    # xi = np.random.normal(scale=std_t, size=n)
    # yi = np.random.normal(scale=std_t, size=n)
    # zi = np.random.normal(scale=std_a, size=n)
    xlist = linspace(-2 * w0, 2 * w0, 1000)
    #print(std_t)
    ps = trapt_p_dist(xlist, std_t, w0)
    xi = random.choice(xlist, n, p=ps / sum(ps))
    yi = random.choice(xlist, n, p=ps / sum(ps))
    zlist = linspace(-2 * rayleigh, 2 * rayleigh, 1000)
    ps = trapz_p_dist(zlist, std_a, rayleigh)
    zi = random.choice(zlist, n, p=ps / sum(ps))

    # when <v(0)> is non-zero (eg from PGC in non-zero B field)
    vx0, vy0, vz0 = v0

    vxi = random.normal(loc=vx0, scale=std_v, size=n)
    vyi = random.normal(loc=vy0, scale=std_v, size=n)
    vzi = random.normal(loc=vz0, scale=std_v, size=n)

    if scramble:
        xi, vxi = dephase_SHO(xi, vxi, freq_t)
        yi, vyi = dephase_SHO(yi, vyi, freq_t)
        zi, vzi = dephase_SHO(zi, vzi, freq_a)

    return xi, yi, zi, vxi, vyi, vzi

def drop_recapture_MC(
        t: itar,
        temperature: float,
        p: float,
        w0: float,
        alpha_0: float = alpha_0_938,
        rayleigh: float = None,
        m: float = mcs,
        wl: float = 938e-9,
        gravity: bool = True,
        n: int = 1000,
        v0: List = None,
        scramble: bool = True
) -> Tuple[itar, itar]:
    """
    Performs a monte-carlo (MC) simulation of a single atom trapped in an ODT that is pulsed off for a time(s)
    t. Computes the probability of the atom to be re-captured once the ODT is turned back on.
    Args:
        t: drop time(s) of the drop-recapture experiment to be sampled (s)
        temperature: temperature of atom (K)
        p: power of the ODT beam (W)
        w0: waist of the ODT beam (m)
        alpha_0: atom polarizability at ODT wavelength. Default is alpha_0 for Cs at 938nm. (Jm^2/V^2)
        rayleigh: rayleigh range of the ODT beam. Computed from w0 and wl if not provided. (m)
        m: mass of the trapped atom. Default is mass of cesium (kg)
        wl: wavelength of trapping light. Default is 938nm (m)
        gravity: if True, gravity is considered in model evolution, otherwise it is ignored.
        n: number of times to repeat the MC simulation. Default if 1000
        v0: default [0,0,0]. If provided and non-zero the Maxwell-Boltzmann distribution for velocities will
            be centered around v0[i] instead of zero.

    Returns:
        retention, retention_std: recapture probability of atom after ODT is dropped for time(s) t, and statistical
            uncertainty in the recapture probability. Computed from the shot-noise limit retention(1-retention)/sqrt(n)
    """
    if rayleigh is None:
        rayleigh = zr(w0, wl)

    # print(w0,rayleigh)
    freq_t = trap_frequency_t(p, w0, alpha_0)
    freq_a = trap_frequency_a(p, w0, alpha_0, rayleigh=rayleigh)
    # print(freq_a/tau,freq_t/tau)

    std_t = sigma_x_boltz_SHO(temperature, freq_t)
    std_a = sigma_x_boltz_SHO(temperature, freq_a)
    std_v = sigma_v_boltz(temperature)
    # print(std_t,std_a,std_v)


    # establish initial conditions
    # xi = np.random.normal(scale=std_t, size=n)
    # yi = np.random.normal(scale=std_t, size=n)
    # zi = np.random.normal(scale=std_a, size=n)

    xi, yi, zi, vxi, vyi, vzi = SHO_ic(temperature, p, w0, alpha_0, rayleigh, wl, n, v0, scramble=scramble)

    Uk = 1 / 2 * m * (vxi ** 2 + vyi ** 2 + vzi ** 2) / kb  # kinetic energy in K
    Upi = Ixyz(xi, yi, zi, w0, wl, -trap_depth(p, w0, alpha_0))  # potential energy in K
    Ui = Uk + Upi
    #print(trap_depth(p,w0,alpha_0))

    # remove atoms that would not be trapped initially
    bad_inds = np.argwhere(Ui > 0)
    Ui[bad_inds] = np.NaN

    try:
        retention = np.zeros(len(t))
        ret_std = np.zeros(retention.shape)
        for i, dt in enumerate(t):
            xf = xi+vxi*dt  #m
            yf = yi+vyi*dt-0.5*gravity*g*dt**2  #m
            zf = zi+vzi*dt  #m
            Upf = Ixyz(xf, yf, zf, w0, wl, -trap_depth(p, w0, alpha_0))
            #print(Ixyz(linspace(-w0,w0,100), 0,0, w0, wl, -trap_depth(p, w0, alpha_0)))
            Uf = Upf+Uk
            Uf[bad_inds] = np.NaN

            keep = Uf<0
            retention[i] = sum(keep)/(n-len(bad_inds))
            ret_std[i] = retention[i]*(1-retention[i])/sqrt(n-len(bad_inds))
            #print(len(bad_inds), sum(keep))
            #print("Upf (uK)",Upf*1e6)
            #print('-'*30)
            #print("Upi (uK)",Upi*1e6)
            #print('-'*30)
            #print("Uki (uK)",Uk*1e6)
    except TypeError:

        xf = xi+vxi*t  #m
        yf = yi+vyi*t-0.5*g*gravity*t**2  #m
        zf = zi+vzi*t  #m
        Upf = Ixyz(xf, yf, zf, w0, wl, -trap_depth(p, w0, alpha_0))

        Uf = Upf+Uk
        Uf[bad_inds] = np.NaN

        keep = Uf < 0
        retention = sum(keep)/(n-len(bad_inds))
        ret_std = retention*(1-retention)/sqrt(n-len(bad_inds))

    return retention, ret_std


def drop_recapture_MC_chiSquare(
        t: itar,
        experimental_data: ndarray,
        experiment_data_errors: ndarray,
        baseline_ret: float,
        temperature: float,
        p: float,
        w0: float,
        alpha_0: float = alpha_0_938,
        rayleigh: float = None,
        m: float = mcs,
        wl: float = 938e-9,
        gravity: bool = True,
        n: int = 1000,
        v0: List = None,
) -> Tuple[itar, itar, itar]:
    """
    Performs a monte-carlo (MC) simulation of a single atom trapped in an ODT that is pulsed off for a time(s)
    t. Computes the probability of the atom to be re-captured once the ODT is turned back on.
    Args:
        t: drop time(s) of the drop-recapture experiment to be sampled (s)
        experimental_data: Experimentally measured retention through Drop Recapture (fractions)
        experiment_data_errors: Uncertainty in measured retention (fractions)
        baseline_ret: Baseline retention with no drop (fraction)
        temperature: temperature of atom (K)
        p: power of the ODT beam (W)
        w0: waist of the ODT beam (m)
        alpha_0: atom polarizability at ODT wavelength. Default is alpha_0 for Cs at 938nm. (Jm^2/V^2)
        rayleigh: rayleigh range of the ODT beam. Computed from w0 and wl if not provided. (m)
        m: mass of the trapped atom. Default is mass of cesium (kg)
        wl: wavelength of trapping light. Default is 938nm (m)
        gravity: if True, gravity is considered in model evolution, otherwise it is ignored.
        n: number of times to repeat the MC simulation. Default if 1000
        v0: default [0,0,0]. If provided and non-zero the Maxwell-Boltzmann distribution for velocities will
            be centered around v0[i] instead of zero.

    Returns:
        retention, retention_std, chi_square: recapture probability of atom after ODT is dropped for time(s) t, statistical
            uncertainty in the recapture probability (Computed from the shot-noise limit retention(1-retention)/sqrt(n)),
            chi_square of experimental data compared to simultated data."""
    retention, ret_std = drop_recapture_MC(t=t, temperature=temperature, p=p, w0=w0, alpha_0=alpha_0, rayleigh=rayleigh,
                                           m=m, wl=wl, gravity=gravity, n=n, v0=v0)
    retention = baseline_ret * retention
    residuals = experimental_data - retention
    total_sigma = np.sqrt(experiment_data_errors ** 2 + ret_std ** 2)
    chi_square = np.sum((residuals / total_sigma) ** 2)
    return (retention, ret_std, chi_square)


def get_temp(
        iterations: Iterations,
        retention: itar,
        ret_std: itar,
        max_retention: float,
        p: float,
        w0: float,
        alpha_0: float = alpha_0_938,
        rayleigh: float = None,
        m: float = mcs,
        wl: float = 938e-9,
        gravity: bool = True,
        n: int = 1000,
        temp_list: itar = None
) -> Tuple[float,float,itar]:
    """

    Args:
        iterations: Experiment iterations object
        retention: Experimentally measured retention through Drop Recapture (fractions)
        ret_std: Uncertainty in measured retention (fractions)
        max_retention: Baseline retention with no drop (fraction)
        p: Power of the ODT beam (W)
        w0: Waist of the ODT beam (m)
        alpha_0: atom polarizability at ODT wavelength. Default is alpha_0 for Cs at 938nm. (Jm^2/V^2)
        rayleigh: rayleigh range of the ODT beam. Computed from w0 and wl if not provided. (m)
        m: mass of the trapped atom. Default is mass of cesium (kg)
        wl: wavelength of trapping light. Default is 938nm (m)
        gravity: if True, gravity is considered in model evolution, otherwise it is ignored.
        n: number of times to repeat the MC simulation. Default if 1000
        temp_list: ndarray of temperatures to sample for best fit (uK)

    Returns:
        T, dT, Xs:
            T: best fit Temperature (uK)
            dT: uncertainty in T (uK)
            Xs: List of Chi-Squared Values sampled at temp_list
    """

    def fit_function(x, a, b, c, d, e, x0):
        return(a*(x-x0)**5 + b*(x-x0)**4 + c*(x-x0)**3 + d*(x-x0)**2 + e)

    if temp_list is None:
        temp_list = arange(1, 100, 1)  # 0 - 100uK in 1uK steps

    drop_times = iterations.independent_variables[iterations.ivars[0]]*10**(-6)
    chi_square_vals = []

    for temp in temp_list:
        sim_retention, sim_ret_std, sim_chi_square = drop_recapture_MC_chiSquare(t=drop_times,
                                                                                 experimental_data=retention,
                                                                                 experiment_data_errors=ret_std,
                                                                                 baseline_ret=max_retention,
                                                                                 temperature=temp*10**(-6),
                                                                                 p=p,
                                                                                 w0=w0,
                                                                                 alpha_0=alpha_0,
                                                                                 rayleigh=rayleigh,
                                                                                 m=m,
                                                                                 wl=wl,
                                                                                 gravity=gravity,
                                                                                 n=n)
        chi_square_vals.append(sim_chi_square)

    temp_guess = float(temp_list[chi_square_vals == np.amin(chi_square_vals)])

    popt, pcov = curve_fit(fit_function, temp_list, chi_square_vals, maxfev=5000, p0=[0,0,0,0,0,temp_guess])
    uncertainX0 = np.sqrt((1/popt[3]) + np.diag(pcov)[5])
    return (popt[5], uncertainX0, chi_square_vals, popt)
