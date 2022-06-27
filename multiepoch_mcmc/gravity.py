import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.optimize import brentq

# constants in [km,Msun] units
G = const.G.to(u.km**3/(u.Msun*u.s**2)).value
c = const.c.to(u.km/u.s).value

# conversion factors
g_to_cm = 1e14  # from [1e14 cm/s^2] to [cm/s^2]
g_to_km = 1e9   # from [1e14 cm/s^2] to [km/s^2]


# ===============================================================
#                      Newtonian <---> GR
# ===============================================================
def gr_corrections(r, m, phi=1.0):
    """Returns GR correction factors given Newtonian radius and mass
        Ref: Eq. B5, Keek & Heger 2011

    Returns: xi, redshift
        xi: radius ratio (R_GR / R_NW)
        redshift: (1+z) factor

    parameters
    ----------
    r : float [km]
        Newtonian radius
    m : float [Msun]
        Newtonian mass
    phi : float
        mass ratio (M_GR / M_NW)
    """
    xi = get_xi(r=r, m=m, phi=phi)
    redshift = xi**2 / phi

    return xi, redshift


def get_xi(r, m, phi):
    """Returns radius ratio (R_GR / R_NW) given Newtonian mass, radius, and phi

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    phi : float
        mass ratio (M_GR / M_NW)
    """
    zeta = get_zeta(r=r, m=m)

    b = (9 * zeta**2 * phi**4 + np.sqrt(3) * phi**3 * np.sqrt(16 + 27 * zeta**4 * phi**2))**(1/3)
    a = (2 / 9)**(1 / 3) * (b**2 / phi**2 - 2 * 6**(1 / 3)) / (b * zeta**2)
    xi = (zeta * phi / 2) * (1 + np.sqrt(1 - a) + np.sqrt(2 + a + 2 / np.sqrt(1 - a)))

    return xi


def get_redshift(r, m):
    """Returns redshift (1+z) for given radius and mass

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    """
    zeta = get_zeta(r=r, m=m)
    redshift = 1 / np.sqrt(1 - 2*zeta)

    return redshift


def get_zeta(r, m):
    """Returns zeta factor (GM/Rc^2) for given radius and mass

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    """
    zeta = (G * m) / (r * c**2)
    return zeta


# ===============================================================
#                   g, redshift ---> mass, radius
# ===============================================================
def get_mass_radius(g, redshift):
    """Return GR mass and radius for given gravity and redshift

    g : float
        gravitational acceleration [10^14 cm/s^2]
    redshift : float
        (1+z) factor
    """
    r = get_radius(g=g, redshift=redshift)
    m = get_mass(g=g, redshift=redshift)
    return m, r


def get_radius(g, redshift):
    """Return GR NS radius for given gravity and redshift

    Ref: Eq. B24, Keek & Heger (2011)

    Parameters
    ----------
    g : float
        gravitational acceleration [10^14 cm/s^2]
    redshift : float
        (1+z) factor
    """
    z = redshift - 1
    g *= g_to_km

    r = (c ** 2 * z * (z + 2)) / (2 * g * redshift)

    return r


def get_mass(g, redshift):
    """Return GR NS mass for given gravity and redshift
         Eq. B24, Keek & Heger (2011)

    Parameters
    ----------
    g : float
        gravitational acceleration [10^14 cm/s^2]
    redshift : float
        (1+z) factor
    """
    z = redshift - 1
    g *= g_to_km

    m = (c**4 * z**2 * (z + 2)**2) / (4 * G * g * redshift**3)

    return m


# ===============================================================
#                   mass, radius ---> g
# ===============================================================
def get_acceleration_newtonian(r, m):
    """Returns Newtonian gravitational acceleration given radius and mass

    Returns: float [1e14 cm/s^2]

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    """
    g_newton = G * m / (g_to_km * r**2)
    return g_newton


def get_acceleration_gr(r, m):
    """Returns GR gravitational acceleration given radius and mass

    Returns: float [1e14 cm/s^2]

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    """
    redshift = get_redshift(r=r, m=m)
    g_newton = get_acceleration_newtonian(r=r, m=m)
    g_gr = g_newton * redshift

    return g_gr


# ===============================================================
#                   g, mass/radius ---> mass/radius
# ===============================================================
def r_from_g(g, m):
    """Returns radius given gravitational acceleration and mass

    Returns: float [km]

    Parameters
    ----------
    g : float [1e14 cm/s^2]
    m : float [Msun]
    """
    def root(r_root, m_root, g_root):
        return get_acceleration_gr(r=r_root, m=m_root) - g_root

    r = brentq(root, 6, 50, args=(m, g))

    return r


def mass_from_g(g, r):
    """Returns Newtonian mass given acceleration and radius

    Returns: float [Msun]

    Parameters
    ----------
    g : float [1e14 cm/s^2]
    r : float [km]
    """
    g *= g_to_km
    m = g * r**2 / G

    return m


# ===============================================================
#                   mass, radius ---> potential
# ===============================================================
def get_potential_newtonian(r, m):
    """Returns Newtonian gravitational potential given radius and mass

    Returns: float [km^2 / s^2]

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    """
    potential = -G * m / r
    return potential


def get_potential_gr(r, m):
    """Returns GR gravitational potential given mass and radius

    Returns: float [km^2 / s^2]

    Parameters
    ----------
    r : float [km]
    m : float [Msun]
    """
    redshift = get_redshift(r=r, m=m)
    potential = -(redshift - 1) * c**2 / redshift

    return potential
