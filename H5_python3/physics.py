from scipy import constants
from numpy import *

# fundamental constants
kb = constants.Boltzmann
hb = constants.hbar
c = constants.c
eps = constants.epsilon_0

# engineering constants
g = 9.81  # m/s**2
gu = g * 1e-6  # um/us**2

# Properties of cesium
mcs = 2.2069393e-25  # kg

## Polarizabilities
alpha_0_938_CGS = 429  # angstrom^3
alpha_0_938 = alpha_0_938_CGS * 1e-30 * 4 * pi * eps  # Jm^2/V^2

# Common equations

## Gaussian Beams
def I0(P, width):
    """
    Max intensity of a gaussian beam
    Args:
        P: power in gaussian beam (W)
        width: width of gaussian beam (m)

    Returns:
        I0: Peak intensity of a gaussian beam (W/m^2)
    """
    return 2 * P / pi / width ** 2


def Esquared(intensity):
    """
    Square of electric field oscillation amplitude in vacuum
    Args:
        intensity: intensity of a beam (W/m^2)
    Returns:
        E^2(I) (V^2/m^2)
    """
    return 2 * intensity / c / eps


def zr(w0, wl):
    """
    Rayleigh range of gaussian beam
    Args:
        w0: width of gaussian beam (m)
        wl: wavelength of gaussian beam (m)
    Returns:
        zr: Rayleigh range of gaussian beam (m)
    """
    return pi * w0 ** 2 / wl


def wz(z, w0, wl):
    """
    Width of a gaussian beam at position z along axial direction
    Args:
        z: z-position on which width is being estimated (m)
        w0: waist of the beam (m)
        wl: wavelength of beam (m)

    Returns:
        w(z): width of beam at given z position (m)
    """
    return w0*sqrt(1+z/zr(w0, wl))


def Ixyz(x, y, z, w0, wl, Im):
    """
    Intensity of a gaussian beam at positions x,y,z (relaive to center of beam waist)
    Args:
        x: x-position (m)
        y: y-position (m)
        z: z-position (m)
        p: Power in beam (W)
        w0: waist of beam (m)
        wl: wavelength of beam (m)
    Returns:
        Intensity at x,y,z positions provided (W/m^2)
    """
    width = wz(z, w0, wl)
    return Im * (w0 / width) ** 2 * exp(-2 * (x ** 2 + y ** 2) / width ** 2)


## Red optical dipole traps (ODT)
def trap_depth(p: float, w0: float, alpha0: float = alpha_0_938):
    """
    Computes depth of a red ODT
    Args:
        p : power in ODT beam (W)
        w0 : waist of ODT beam (m)
        alpha0 : polarizability of ODT (Jm^2/V^2)

    Returns:
        Ut: Depth of ODT (K)
    """
    return alpha0/4*Esquared(I0(p, w0))/kb


def trap_frequency_t(p: float, w0: float, alpha0: float = alpha_0_938):
    """
    Computes transverse trap frequency for red ODT
    Args:
        p : power in ODT beam (W)
        w0 : waist of ODT beam (m)
        alpha0 : polarizability of ODT (Jm^2/V^2)

    Returns:
        omega_T : Transverse trap frequency (radians/s)
    """
    return sqrt(4*kb*trap_depth(p, w0, alpha0)/mcs/w0**2)


def trap_frequency_a(p: float, w0: float, wl: float = 938e-9, alpha0: float = alpha_0_938, rayleigh=None):
    """
    Computes axial trap frequency for red ODT
    Args:
        p : power in ODT beam (W)
        w0 : waist of ODT beam (m)
        wl: wavelength of ODT beam (m)
        alpha0 : polarizability of ODT (Jm^2/V^2)
        rayleigh : rayleigh range of ODT beam. If not provided computed automatically (m)

    Returns:
        omega_A: Axial trap frequency (radians/s)
    """
    if rayleigh is None:
        rayleigh = zr(w0, wl)
    return sqrt(2*kb*trap_depth(p, w0, alpha0)/mcs/rayleigh**2)


def sigma_v_boltz(temperature: float):
    """
    Standard deviation of a Maxwell-Boltzmann velocity distribution of cesium atoms

    Args:
        temperature: temperature of the atoms (K)

    Returns:
        sigma_v: std of MB distribution (m/s)
    """
    return sqrt(kb*temperature/mcs)


def sigma_x_boltz_SHO(temperature: float, osc_frequency: float):
    """
    Standard deviation of a Maxwell-Boltzmann position distribution of cesium atoms in a simple
    harmonic oscillator (SHO)

    Args:
        temperature: temperature of the atoms (K)
        osc_frequency: oscillation frequency of the SHO (radians/s)

    Returns:
        sigma_x: std of MB distribution (m)
    """
    return sqrt(kb*temperature/mcs/osc_frequency**2)

def trapt_p_dist(x, std, w0):
    """
    Boltzmann based probability distribution of atoms in the trap, taking approximation of FORT potential to (x/w0)^6
    Args:
        x: position in the trap (m)
        std: standard deviation of SHO approximated trap (m)
        w0: waist of the trap (m)
    Returns:
        probability of x position being occupied by a thermal atom (not normalized!!)
    """
    # print(std,w0)
    a = 1/w0**2
    b = 2/3/w0**4
    # print(a,b)
    px = exp(-1/2*((x/std)**2)*(1-a*x**2+b*x**4))
    # print(px)
    return px

def trapz_p_dist(x, std, rayleigh):
    """
    Boltzmann based probability distribution of atoms in the trap, taking approximation of FORT potential to (x/w0)^6
    Args:
        x: position in the trap (m)
        std: standard deviation of SHO approximated trap (m)
        rayleigh: waist of the trap (m)
    Returns:
        probability of x position being occupied by a thermal atom (not normalized!!)
    """
    a = 1/rayleigh**2
    b = 1/rayleigh**4
    pz = exp(-(1/2*(x/std)**2)*(1-a*x**2+b*x**4))
    return pz
