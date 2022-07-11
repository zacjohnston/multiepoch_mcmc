import numpy as np
import astropy.units as u

from multiepoch_mcmc import accretion, gravity, config
from multiepoch_mcmc.grid_interpolator import GridInterpolator


class BurstModel:
    """Class to model bursts given a set of parameters
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
        self._epoch_params = self._config['interp']['params']
        self._epoch_unique = self._config['keys']['epoch_unique']
        self._n_epochs = len(self._config['obs']['epochs'])

        self.bvars = self._config['keys']['bvars']
        self._interp_bvars = self._config['keys']['interp_bvars']
        self._analytic_bvars = self._config['keys']['analytic_bvars']

        self._kepler_radius = self._config['grid']['kepler_radius']
        self._u_frac = self._config['lhood']['u_frac']

        # dynamic variables
        self._x = np.empty(len(self.params))
        self._x_key = dict.fromkeys(self.params)
        self._x_epoch = np.empty((self._n_epochs, len(self._epoch_params)))
        self._terms = {}
        self._interp_local = np.empty([self._n_epochs, 2*len(self._interp_bvars)])
        self._analytic_local = np.empty([self._n_epochs, 2*len(self._analytic_bvars)])
        self._y_local = np.empty([self._n_epochs, 2*len(self.bvars)])
        self._y_observer = np.empty_like(self._y_local)

        self._interpolator = GridInterpolator(file=self._config['interp']['file'],
                                              params=self._epoch_params,
                                              bvars=self._config['interp']['bvars'],
                                              reconstruct=False,
                                              ).interpolate

    # ===============================================================
    #                      Burst Sampling
    # ===============================================================
    def sample(self, x):
        """Returns the predicted observables for given coordinates

        Returns: [n_epochs, n_bvars]

        Parameters
        ----------
        x : 1Darray
            sample coordinates
        """
        self._x = x
        self._fill_x_key()
        self._get_x_epoch()
        self._get_all_terms()

        self._get_y_local()
        self._get_y_observer()

        return self._y_observer

    def _get_y_local(self):
        """Calculates model values for given coordinates

        Assumes the following have already been executed:
            - self._fill_x_key()
            - self._get_x_epoch()
            - self._get_all_terms()

        Returns: [n_epochs, n_bvars]
        """
        self._interpolate()
        self._get_analytic()

        self._y_local = np.concatenate([self._interp_local, self._analytic_local], axis=1)

    def _interpolate(self):
        """Interpolates burst properties for N epochs

        Assumes the following have already been executed:
            - self._fill_x_key()
            - self._get_x_epoch()
        """
        self._interp_local = self._interpolator(x=self._x_epoch)

    def _get_analytic(self):
        """Calculates analytic burst properties

        Assumes the following have already been executed:
            - self._fill_x_key()
            - self._get_x_epoch()
            - self._get_all_terms()
        """
        for i, bvar in enumerate(self._analytic_bvars):
            idx = 2 * i
            self._analytic_local[:, idx] = self._terms[bvar]
            self._analytic_local[:, idx+1] = self._terms[bvar] * self._u_frac[bvar]

    def _get_y_observer(self):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame

        Assumes the following have already been executed:
            - self._fill_x_key()
            - self._get_x_epoch()
            - self._get_all_terms()
            - self._get_y_local()

        Returns: [n_epochs, n_bvars]
        """
        for i, bvar in enumerate(self.bvars):
            i0 = 2 * i
            i1 = i0 + 2

            values = self._y_local[:, i0:i1]
            self._y_observer[:, i0:i1] = values * self._terms['shift_factor'][bvar]

    # ===============================================================
    #                      Sample coordinates
    # ===============================================================
    def _fill_x_key(self):
        """Fills dictionary of sample coordinates
        """
        for i, key in enumerate(self.params):
            self._x_key[key] = self._x[i]

    def _get_x_epoch(self):
        """Reshapes sample coordinates into epoch array: [n_epochs, n_epoch_params]

        Assumes the following have already been executed:
            - self._fill_x_key()
        """
        for i in range(self._n_epochs):
            for j, key in enumerate(self._epoch_params):
                if key in self._epoch_unique:
                    key = f'{key}{i+1}'

                self._x_epoch[i, j] = self._x_key[key]

    # ===============================================================
    #                      Analytic terms
    # ===============================================================
    def _get_all_terms(self):
        """Calculate all derived/analytic terms

        Assumes the following have already been executed:
            - self._fill_x_key()
            - self._get_x_epoch()
        """
        self._get_gravity_terms()
        self._get_conversion_terms()
        self._get_analytic_terms()

    def _get_gravity_terms(self):
        """Calculates gravity terms

        Assumes the following have already been executed:
            - self._fill_x_key()
            - self._get_x_epoch()
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
            - self._fill_x_key()
            - self._get_x_epoch()
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
            - self._fill_x_key()
            - self._get_x_epoch()
            - self._get_gravity_terms()
            - self._get_conversion_terms()
        """
        # Note: actually luminosity until converted
        mdot = self._x_epoch[:, self._epoch_params.index('mdot')]
        self._terms['fper'] = mdot * self._terms['mdot_to_lum']

        # Note: actually luminosity until converted
        self._terms['fedd'] = accretion.edd_lum_newt(m_nw=self._terms['mass_nw'],
                                                     x=self._x_key['x'])
