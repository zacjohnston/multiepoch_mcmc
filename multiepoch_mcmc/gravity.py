import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.optimize import brentq

# Constants in cgs units
G = const.G.to(u.cm**3/(u.g*u.s**2))
c = const.c.to(u.cm/u.s)
Msun_in_g = const.M_sun.to(u.g)


def apply_units(r, m):
    """Return get_radius and get_mass with units (cm, g)

    Assumes get_radius given in km, get_mass given in Msun
    """
    r = (r * u.km).to(u.cm)
    m = m * Msun_in_g
    return r, m


def get_redshift(r, m):
    """Returns redshift (1+z) for given get_radius and get_mass (assuming GR)
    """
    zeta = get_zeta(r=r, m=m)
    redshift = 1 / np.sqrt(1 - 2*zeta)
    return redshift


def get_zeta(r, m):
    """Returns zeta factor (GM/Rc^2) for given get_radius and get_mass
    """
    r_u, m_u = apply_units(r=r, m=m)
    zeta = (G * m_u) / (r_u * c**2)

    if True in zeta >= 0.5:
        raise ValueError(f'R, M ({r:.2f}, {m:.2f}) returns zeta >= 0.5')

    return np.array(zeta)


def get_mass_radius(g, redshift):
    """Return GR get_mass and get_radius for given gravity and redshift

    g : gravitational acceleration
    redshift : (1+z) redshift factor
    """
    r = get_radius(g=g, redshift=redshift)
    m = get_mass(g=g, redshift=redshift)
    return m, r


def get_radius(g, redshift):
    """Return GR NS radius for given gravity and redshift
             Eq. B24, Keek & Heger (2011)

        g : flt
            gravitational acceleration
        redshift : flt
            (1+z) redshift factor
        """
    z = redshift - 1
    r_u = (c ** 2 * z * (z + 2)) / (2 * g * redshift)
    return r_u.to(u.km)


def get_mass(g, redshift):
    """Return GR NS mass for given gravity and redshift
         Eq. B24, Keek & Heger (2011)

    g : gravitational acceleration
    redshift : (1+z) redshift factor
    """
    z = redshift - 1
    m_u = (c ** 4 * z ** 2 * (z + 2) ** 2) / (4 * G * g * redshift ** 3)
    return m_u.to(u.M_sun)


def get_accelerations(r, m):
    """Returns both gravitational accelerations (Newtonian, GR), given R and M
    """
    g_newton = get_acceleration_newtonian(r=r, m=m)
    g_gr = get_acceleration_gr(r=r, m=m)
    return g_newton, g_gr


def get_acceleration_newtonian(r, m):
    """Returns gravitational accelerations (Newtonian), given R and M
    """
    r_u, m_u = apply_units(r=r, m=m)
    g_newton = G*m_u/r_u**2
    return g_newton


def get_acceleration_gr(r, m):
    """Returns gravitational accelerations (GR), given R and M
    """
    redshift = get_redshift(r=r, m=m)
    g_newton = get_acceleration_newtonian(r=r, m=m)
    g_gr = g_newton * redshift
    return g_gr


def inverse_acceleration(g, m=None, r=None):
    """Returns R or M, given g and one of R or M
    """
    # TODO: solve for m
    def root(r_root, m_root, g_root):
        return get_acceleration_gr(r=r_root, m=m_root).value - g_root.value

    if (m is None) and (r is None):
        print('ERROR: need to specify one of m or r')
    if (m is not None) and (r is not None):
        print('Error: can only specify one of m or r')

    g *= 1e14 * u.cm/u.s/u.s

    if r is None:
        r = brentq(root, 6, 50, args=(m, g))
        return r


def plot_g():
    """Plots g=constant curves against R, M
    """
    g_list = [1.06, 1.33, 2.1, 2.66, 3.45, 4.25]
    m_list = np.linspace(1, 2, 50)
    r_list = np.zeros(50)

    fig, ax = plt.subplots()

    for g in g_list:
        for i, m in enumerate(m_list):
            r_list[i] = inverse_acceleration(g=g, m=m)

        ax.plot(m_list, r_list, label=f'{g:.2f}')

    ax.set_xlabel('Mass (Msun)')
    ax.set_ylabel('Radius (km)')
    ax.legend()
    plt.show(block=False)


def gr_corrections(r, m, phi=1.0, verbose=False):
    """Returns GR correction factors (xi, 1+z) given Newtonian R, M
        Ref: Eq. B5, Keek & Heger 2011

    parameters
    ----------
    m : flt
        Newtonian get_mass (Msol) (i.e. Kepler frame)
    r   : flt
        Newtonian get_radius (km)
    phi : flt
        Ratio of GR get_mass to Newtonian get_mass: M_GR / M_NW
        (NOTE: unrelated to grav potential phi)
    verbose : bool
    """
    zeta = get_zeta(r=r, m=m)

    b = (9 * zeta**2 * phi**4 + np.sqrt(3) * phi**3 * np.sqrt(16 + 27 * zeta**4 * phi**2))**(1/3)
    a = (2 / 9)**(1 / 3) * (b**2 / phi**2 - 2 * 6**(1 / 3)) / (b * zeta**2)
    xi = (zeta * phi / 2) * (1 + np.sqrt(1 - a) + np.sqrt(2 + a + 2 / np.sqrt(1 - a)))

    redshift = xi**2 / phi    # NOTE: xi is unrelated to anisotropy factors xi_b, xi_p

    if verbose:
        print(f'Using R={r:.3f}, M={m}, M_GR={m*phi}:')
        print(f'    R_GR = {r*xi:.2f} km')
        print(f'(1+z)_GR = {redshift:.3f}')
    return xi, redshift


def get_potential_newtonian(r, m):
    """Returns gravitational potentials (phi) given R and M (Newton)
    """
    r_u, m_u = apply_units(r=r, m=m)
    phi_newton = -G*m_u/r_u
    return phi_newton


def get_potential_gr(r=None, m=None, redshift=None):
    """Returns gravitational potentials (phi) given R and M (GR)
    """
    if redshift is None:
        if None in [r, m]:
            raise ValueError('Must provide either redshift, or both r and m')
        redshift = get_redshift(r=r, m=m)

    phi_gr = -(redshift-1)*c**2 / redshift
    return phi_gr


def get_potentials(r, m):
    """Returns both gravitational potentials (phi) given R and M (Newtonian, GR)
    """
    phi_newton = get_potential_newtonian(r=r, m=m)
    phi_gr = get_potential_gr(r=r, m=m)
    return phi_newton, phi_gr


def gravity_summary(r, m):
    """Prints summary gravitational properties given R, M
    """
    redshift = get_redshift(r=r, m=m)
    phi_newton, phi_gr = get_potentials(r=r, m=m)
    g_newton, g_gr = get_accelerations(r=r, m=m)

    print('R (km),  M (Msun)')
    print(f'{r:.2f},   {m:.2f}')

    print('g (Newtonian)')
    print(f'{g_newton:.3e}')

    print('g (GR)')
    print(f'{g_gr:.3e}')

    print('(1+z) (GR)')
    print(f'{redshift:.3f}')

    print('potential (Newtonian, erg/g)')
    print(f'{phi_newton:.3e}')

    print('potential (GR, erg/g)')
    print(f'{phi_gr:.3e}')

    return g_newton, g_gr, phi_newton, phi_gr
