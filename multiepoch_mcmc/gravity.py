import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.optimize import brentq

# constants in [km,Msun] units
G = const.G.to(u.km**3/(u.Msun*u.s**2)).value
c = const.c.to(u.km/u.s).value

# conversion factors
km_to_cm = 1e5
g_to_cm = 1e14                # from [1e14 cm/s^2] to [cm/s^2]
g_to_km = g_to_cm / km_to_cm  # from [1e14 cm/s^2] to [km/s^2]


# ===============================================================
#                      Newtonian <---> GR
# ===============================================================
#     Refs:
#         - Appendix B, Keek & Heger (2011) [arxiv:1110.2172]
#         - Sec 2.3, Johnston (2020) [arxiv:2004.00012]
# ===============================================================
def get_redshift(r_gr, m_gr):
    """Returns redshift (1+z) for given radius and mass

    Parameters
    ----------
    r_gr : float
        GR radius [km]
    m_gr : float
        GR mass [Msun]
    """
    zeta = get_zeta(r=r_gr, m=m_gr)
    redshift = 1 / np.sqrt(1 - 2*zeta)

    return redshift


def get_zeta(r, m):
    """Returns zeta factor (GM/Rc^2) for given radius and mass

    Parameters
    ----------
    r : float
        radius [km]
    m : float
        mass [Msun]
    """
    zeta = (G * m) / (r * c**2)
    return zeta


def redshift_from_xi_phi(xi, phi):
    """Returns redshift given mass and radius ratios

    See: Eq B8, Keek & Heger (2011) [arxiv:1110.2172]

    Returns: float
        (1+z) factor

    Parameters
    ----------
    xi : float
       radius ratio (R_GR / R_NW)
    phi : float
        mass ratio (M_GR / M_NW)
    """
    redshift = xi**2 / phi
    return redshift


def get_xi(r_nw, m_nw, phi):
    """Returns radius ratio (R_GR / R_NW) given Newtonian mass, radius, and phi

    See: Eq B5, Keek & Heger (2011) [arxiv:1110.2172]

    Returns: float

    Parameters
    ----------
    r_nw : float
        Newtonian radius [km]
    m_nw : float
        Newtonian mass [Msun]
    phi : float
        mass ratio (M_GR / M_NW)
    """
    zeta = get_zeta(r=r_nw, m=m_nw)

    b = (9 * zeta**2 * phi**4 + np.sqrt(3) * phi**3 * np.sqrt(16 + 27 * zeta**4 * phi**2))**(1/3)
    a = (2 / 9)**(1 / 3) * (b**2 / phi**2 - 2 * 6**(1 / 3)) / (b * zeta**2)

    xi = (zeta * phi / 2) * (1 + np.sqrt(1 - a) + np.sqrt(2 + a + 2 / np.sqrt(1 - a)))

    return xi


def get_phi(r_nw, m_nw, xi):
    """Returns mass ratio (M_GR / M_NW) given Newtonian mass, radius, and xi

    See: Eq B8, Keek & Heger (2011) [arxiv:1110.2172]

    Returns: float

    Parameters
    ----------
    r_nw : float
        Newtonian radius [km]
    m_nw : float
        Newtonian mass [Msun]
    xi : float
       radius ratio (R_GR / R_NW)
    """
    zeta = get_zeta(r=r_nw, m=m_nw)
    phi = zeta * xi**3 * (np.sqrt(1 + 1 / (zeta**2 * xi**2)) - 1)

    return phi


# ===============================================================
#                   g, redshift ---> mass, radius
# ===============================================================
def get_mass_radius(g_gr, redshift):
    """Return GR mass and radius for given gravity and redshift

    Returns, m_gr, r_gr

    g_gr : float
        GR gravitational acceleration [10^14 cm/s^2]
    redshift : float
        (1+z) factor
    """
    r_gr = get_radius(g_gr=g_gr, redshift=redshift)
    m_gr = get_mass(g_gr=g_gr, redshift=redshift)

    return m_gr, r_gr


def get_radius(g_gr, redshift):
    """Return GR NS radius for given gravity and redshift

    Ref: Eq. B24, Keek & Heger (2011)

    Parameters
    ----------
    g_gr : float
        gravitational acceleration [10^14 cm/s^2]
    redshift : float
        (1+z) factor
    """
    z = redshift - 1
    g_gr *= g_to_km

    r_gr = (c ** 2 * z * (z + 2)) / (2 * g_gr * redshift)

    return r_gr


def get_mass(g_gr, redshift):
    """Return GR NS mass for given gravity and redshift
         Eq. B24, Keek & Heger (2011)

    Parameters
    ----------
    g_gr : float
        GR gravitational acceleration [10^14 cm/s^2]
    redshift : float
        (1+z) factor
    """
    z = redshift - 1
    g_gr *= g_to_km

    m_gr = (c**4 * z**2 * (z + 2)**2) / (4 * G * g_gr * redshift**3)

    return m_gr


# ===============================================================
#                   mass, radius ---> g
# ===============================================================
def get_acceleration_newtonian(r_nw, m_nw):
    """Returns Newtonian gravitational acceleration given radius and mass

    Returns: float [1e14 cm/s^2]

    Parameters
    ----------
    r_nw : float
        Newtonian radius [km]
    m_nw : float
        Newtonian mass [Msun]
    """
    g_nw = G * m_nw / (g_to_km * r_nw**2)
    return g_nw


def get_acceleration_gr(r_gr, m_gr):
    """Returns GR gravitational acceleration given radius and mass

    Returns: float [1e14 cm/s^2]

    Parameters
    ----------
    r_gr : float
        GR radius [km]
    m_gr : float
        GR mass [Msun]
    """
    redshift = get_redshift(r_gr=r_gr, m_gr=m_gr)
    g_newton = get_acceleration_newtonian(r_nw=r_gr, m_nw=m_gr)
    g_gr = g_newton * redshift

    return g_gr


# ===============================================================
#                   g, mass/radius ---> mass/radius
# ===============================================================
def r_from_g(g_gr, m_gr):
    """Returns GR radius given gravitational acceleration and mass

    Returns: float [km]

    Parameters
    ----------
    g_gr : float
        GR gravitational acceleration [1e14 cm/s^2]
    m_gr : float
        GR mass [Msun]
    """
    def root(r_root, m_root, g_root):
        return get_acceleration_gr(r_gr=r_root, m_gr=m_root) - g_root

    r_gr = brentq(root, 6, 50, args=(m_gr, g_gr))

    return r_gr


def mass_from_g(g_nw, r_nw):
    """Returns Newtonian mass given acceleration and radius

    Returns: float [Msun]

    Parameters
    ----------
    g_nw : float
        Newtonian gravitational acceleration [1e14 cm/s^2]
    r_nw : float
        Newtonian radius [km]
    """
    g_nw *= g_to_km
    m_nw = g_nw * r_nw**2 / G

    return m_nw


# ===============================================================
#                   mass, radius ---> potential
# ===============================================================
def get_potential_newtonian(r_nw, m_nw):
    """Returns Newtonian gravitational potential given radius and mass

    Returns: float [erg / g]

    Parameters
    ----------
    r_nw : float
        Newtonian radius [km]
    m_nw : float
        Newtonian mass [Msun]
    """
    pot_nw = -(G * m_nw * km_to_cm**2) / r_nw
    return pot_nw


def get_potential_gr(r_gr, m_gr):
    """Returns GR gravitational potential given mass and radius

    Returns: float [erg / g]

    Parameters
    ----------
    r_gr : float
        GR radius
    m_gr : float
        GR mass [Msun]
    """
    redshift = get_redshift(r_gr=r_gr, m_gr=m_gr)
    pot_gr = potential_from_redshift(redshift=redshift)

    return pot_gr


def potential_from_redshift(redshift):
    """Returns GR gravitational potential given redshift

    Returns: float [erg / g]

    Parameters
    ----------
    redshift : float
        (1+z) factor
    """
    pot_gr = -(redshift - 1) * c**2 * km_to_cm**2 / redshift
    return pot_gr
