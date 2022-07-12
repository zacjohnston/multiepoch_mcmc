import numpy as np
import astropy.units as u

from multiepoch_mcmc import accretion, gravity, config
from multiepoch_mcmc.grid_interpolator import GridInterpolator


class BurstModel:
    """Class to model bursts given a set of parameters

    Attributes
    ----------
    bvars : [str]
        burst variable keys
    params : [str]
        model parameter keys
    system : str
        name of bursting system being modelled

    Methods
    -------
    sample(x)
        Returns modelled burst properties for given sample coordinates
    """
    _mdot_edd = 1.75e-8 * (u.M_sun / u.year).to(u.g / u.s)
    _kpc_to_cm = u.kpc.to(u.cm)

    def __init__(self,
                 system='gs1826',
                 ):
        """
        Parameters
        ----------
        system : str
        """
        self.system = system
        self._config = config.load_config(system=self.system)

        self.params = self._config['keys']['params']
        self._interp_params = self._config['interp']['params']
        self._epoch_unique = self._config['keys']['epoch_unique']
        self._n_epochs = len(self._config['obs']['epochs'])

        self.bvars = self._config['keys']['bvars']
        self._interp_bvars = self._config['keys']['interp_bvars']
        self._analytic_bvars = self._config['keys']['analytic_bvars']

        self._n_params = len(self.params)
        self._n_bvars = len(self.bvars)
        self._n_interp = len(self._interp_bvars)
        self._n_analytic = len(self._analytic_bvars)

        self._kepler_radius = self._config['grid']['kepler_radius']
        self._u_frac = self._config['lhood']['u_frac']

        # dynamic variables
        self._x = np.empty(self._n_params)
        self._x_key = dict.fromkeys(self.params)
        self._x_epoch = np.empty((self._n_epochs, len(self._interp_params)))
        self._terms = {}

        self._y = np.empty([self._n_epochs, self._n_bvars])
        self._u_y = np.empty_like(self._y)

        self._interpolator = GridInterpolator(file=self._config['interp']['file'],
                                              params=self._interp_params,
                                              bvars=self._config['interp']['bvars'],
                                              reconstruct=False,
                                              ).interpolate

    # ===============================================================
    #                      Burst Sampling
    # ===============================================================
    def sample(self, x):
        """Returns the modelled burst properties for given coordinates

        Returns: y, u_y
            - y: burst properties of array shape [n_epochs, n_bvars]
            - u_y: corresponding uncertainites (same shape)


        Parameters
        ----------
        x : 1Darray
            sample coordinates, must match length and order of `params`
        """
        self._unpack_coordinates(x)
        self._get_all_terms()

        self._get_y_local()
        self._shift_to_observer()

        return self._y, self._u_y

    def _get_y_local(self):
        """Calculates model values for given coordinates (in local NS frame)

        Assumes the following have already been executed:
            - self._unpack_coordinates()
            - self._get_all_terms()

        Returns: [n_epochs, n_bvars]
        """
        self._get_interpolated()
        self._get_analytic()

    def _get_interpolated(self):
        """Interpolates burst properties for all epochs

        Assumes the following have already been executed:
            - self._unpack_coordinates()
        """
        y_interp = self._interpolator(x=self._x_epoch)

        self._y[:, :self._n_interp] = y_interp[:, ::2]
        self._u_y[:, :self._n_interp] = y_interp[:, 1::2]

    def _get_analytic(self):
        """Calculates analytic burst properties

        Assumes the following have already been executed:
            - self._unpack_coordinates()
            - self._get_all_terms()
        """
        for i, bvar in enumerate(self._analytic_bvars):
            idx = self._n_interp + i
            self._y[:, idx] = self._terms[bvar]
            self._u_y[:, idx] = self._terms[bvar] * self._u_frac[bvar]

    def _shift_to_observer(self):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame

        Assumes the following have already been executed:
            - self._unpack_coordinates()
            - self._get_all_terms()
            - self._get_y_local()
        """
        for i, bvar in enumerate(self.bvars):
            self._y[:, i] *= self._terms['shift_factor'][bvar]
            self._u_y[:, i] *= self._terms['shift_factor'][bvar]

    # ===============================================================
    #                      Sample coordinates
    # ===============================================================
    def _unpack_coordinates(self, x):
        """Unpacks sample coordinates

        parameters
        ----------
        x : [n_params]
            sample coordinates, must match length and order of `params`
        """
        self._x = x
        self._fill_x_key()
        self._get_x_epoch_array()

    def _fill_x_key(self):
        """Fills dictionary of sample coordinates
        """
        for i, key in enumerate(self.params):
            self._x_key[key] = self._x[i]

    def _get_x_epoch_array(self):
        """Reshapes sample coordinates into epoch array: [n_epochs, n_interp_params]

        Assumes the following have already been executed:
            - self._fill_x_key()
        """
        for i in range(self._n_epochs):
            for j, key in enumerate(self._interp_params):
                if key in self._epoch_unique:
                    key = f'{key}{i+1}'

                self._x_epoch[i, j] = self._x_key[key]

    # ===============================================================
    #                      Analytic terms
    # ===============================================================
    def _get_all_terms(self):
        """Calculate all derived/analytic terms

        Assumes the following have already been executed:
            - self._unpack_coordinates()
        """
        self._get_gravity_terms()
        self._get_conversion_terms()
        self._get_analytic_terms()

    def _get_gravity_terms(self):
        """Calculates gravity terms

        Assumes the following have already been executed:
            - self._unpack_coordinates()
        """
        mass_nw = gravity.mass_from_g(g_nw=self._x_key['g'],
                                      r_nw=self._kepler_radius)

        phi = self._x_key['mass'] / mass_nw

        r_ratio = gravity.get_xi(r_nw=self._kepler_radius,
                                 m_nw=mass_nw,
                                 phi=phi)

        redshift = gravity.redshift_from_xi_phi(phi=phi, xi=r_ratio)

        potential = -gravity.potential_from_redshift(redshift)

        self._terms['mass_nw'] = mass_nw
        self._terms['phi'] = phi
        self._terms['r_ratio'] = r_ratio
        self._terms['redshift'] = redshift
        self._terms['potential'] = potential

    def _get_conversion_terms(self):
        """Calculates unit & frame conversion terms

        Assumes the following have already been executed:
            - self._unpack_coordinates()
            - self._get_gravity_terms()
        """
        burst_lum_to_flux = 4 * np.pi * (self._x_key['d_b'] * self._kpc_to_cm)**2
        pers_lum_to_flux = burst_lum_to_flux * self._x_key['xi_ratio']

        phi = self._terms['phi']
        redshift = self._terms['redshift']

        fluence_factor = phi / burst_lum_to_flux
        burst_factor = phi / (burst_lum_to_flux * redshift)
        pers_factor = phi / (pers_lum_to_flux * redshift)

        self._terms['shift_factor'] = {'rate': 1 / redshift,
                                       'fluence': fluence_factor,
                                       'peak': burst_factor,
                                       'fedd': burst_factor,
                                       'fper': pers_factor,
                                       }

        self._terms['mdot_to_lum'] = self._mdot_edd * self._terms['potential']

    def _get_analytic_terms(self):
        """Calculates analytic terms

        Assumes the following have already been executed:
            - self._unpack_coordinates()
            - self._get_gravity_terms()
            - self._get_conversion_terms()
        """
        # Note: actually luminosity until converted
        mdot = self._x_epoch[:, self._interp_params.index('mdot')]
        self._terms['fper'] = mdot * self._terms['mdot_to_lum']

        # Note: actually luminosity until converted
        self._terms['fedd'] = accretion.edd_lum_newt(m_nw=self._terms['mass_nw'],
                                                     x=self._x_key['x'])
