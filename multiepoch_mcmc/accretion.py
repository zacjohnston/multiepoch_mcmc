import numpy as np
from astropy import units
import astropy.constants as const

from multiepoch_mcmc import gravity


def edd_lum_newt(m_nw, x):
    """Returns Newtonian Eddington luminosity for given mass and composition

    Returns: flt
        spherical Eddington luminosity [erg/s]

    Parameters
    ----------
    m_nw : flt
        newtonian stellar mass [M_sun]
    x : flt
        hydrogen composition [mass fraction]
    """
    l_edd_h = edd_lum_hydrogen(m_nw)

    # correct for hydrogen/helium composition (mass/charge ratio)
    l_edd = l_edd_h * 2 / (x + 1)

    return l_edd


def edd_lum_gr(m_gr, radius, x):
    """Returns GR Eddington luminosity for given mass and composition

    Returns: flt
        spherical Eddington luminosity [erg/s]

    Parameters
    ----------
    m_gr : flt
        GR stellar mass [M_sun]
    radius : flt
        GR stellar radius [km]
    x : flt
        hydrogen composition [mass fraction]
    """
    redshift = gravity.get_redshift(r_gr=radius, m_gr=m_gr)
    l_edd_h = redshift * edd_lum_hydrogen(m_gr)

    # correct for hydrogen/helium composition (mass/charge ratio)
    l_edd = l_edd_h * 2 / (x + 1)

    return l_edd


def edd_lum_hydrogen(m_nw):
    """Returns Newtonian Eddington luminosity for pure hydrogen

    Returns: flt
        spherical Eddington luminosity [erg/s]

    Parameters
    ----------
    m_nw : flt
        Newtonian stellar mass [M_sun]
    """
    constants = 4 * np.pi * const.G * const.m_p * const.c / const.sigma_T

    l_edd_h = constants * (m_nw * units.M_sun)
    l_edd_h = l_edd_h.to(units.erg / units.s).value

    return l_edd_h
