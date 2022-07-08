import os
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const

# pyburst
from multiepoch_mcmc import accretion, gravity, config
from multiepoch_mcmc.grid_interpolator import GridInterpolator


class ZeroLhood(Exception):
    """Escape calculation if likelihood is zero
    """
    pass


class BurstSampler:
    """
    Class for sampling modelled bursts and comparing to observations

    Attributes
    ----------
    bvars : [str]
        burst variable keys
    epochs : [int]
        observation epochs to model
    obs_data : {}
        observed burst variables to compare to
    params : [str]
        model parameter keys
    system : str
        name of bursting system being modelled

    Methods
    -------
    lhood(x)
        Calculates log-likelihood for given sample coordinates
    sample(x)
        Returns the modelled burst variables at given sample coordinates
    compare(model, u_model, bvar)
        Returns log-likelihood for given model-observation comparison
    lnprior(x)
        Returns prior likelihoods for given sample coordinates
    """
    _c = const.c.to(u.cm / u.s).value  # speed of light
    _mdot_edd = 1.75e-8 * (u.M_sun / u.year).to(u.g / u.s)  # eddington accretion rate
    _kpc_to_cm = u.kpc.to(u.cm)
    _zero_lhood = -np.inf

    def __init__(self,
                 system='gs1826',
                 ):
        """
        Parameters
        ----------
        system : str
            name of bursting system
        """
        self.system = system
        self._config = config.load_config(system=self.system)

        self.epochs = self._config['obs']['epochs']
        self._n_epochs = len(self.epochs)

        self.params = self._config['keys']['params']
        self._epoch_params = self._config['interp']['params']
        self._epoch_unique = self._config['keys']['epoch_unique']
        self.n_dim = len(self.params)

        self.bvars = self._config['keys']['bvars']
        self._interp_bvars = self._config['keys']['interp_bvars']
        self._analytic_bvars = self._config['keys']['analytic_bvars']

        self._grid_bounds = self._config['grid']['bounds']
        self.x_start = self._config['grid']['x_start']
        self._kepler_radius = self._config['grid']['kepler_radius']

        self._weights = self._config['lhood']['weights']
        self._u_frac = self._config['lhood']['u_frac']
        self._priors = self._config['lhood']['priors']
        self._obs_table = None
        self.obs_data = None

        self._idxs = {}
        for i, key in enumerate(self.params):
            self._idxs[key] = i

        # dynamic variables
        self._x = np.empty(len(self.params))
        self._x_key = dict.fromkeys(self.params)
        self._x_epoch = np.empty((self._n_epochs, len(self._epoch_params)))
        self._terms = {}
        self._lum_to_flux = dict.fromkeys(['burst', 'pers'])
        self._interp_local = np.empty([self._n_epochs, 2*len(self._interp_bvars)])
        self._analytic_local = np.empty([self._n_epochs, 2*len(self._analytic_bvars)])
        self._y_local = np.empty([self._n_epochs, 2*len(self.bvars)])
        self._y_observer = np.empty_like(self._y_local)

        self._unpack_obs_data()

        self._grid_interpolator = GridInterpolator(file=self._config['interp']['file'],
                                                   params=self._epoch_params,
                                                   bvars=self._config['interp']['bvars'],
                                                   reconstruct=False)

    # ===============================================================
    #                      Setup
    # ===============================================================
    def _unpack_obs_data(self):
        """Unpacks observed burst data from table
        """
        self._load_obs_table()
        self.obs_data = self._obs_table.to_dict(orient='list')

        for key, item in self.obs_data.items():
            self.obs_data[key] = np.array(item)

        # ===== Apply bolometric corrections (cbol) to fper ======
        u_fper_frac = np.sqrt((self.obs_data['u_cbol'] / self.obs_data['cbol']) ** 2
                              + (self.obs_data['u_fper'] / self.obs_data['fper']) ** 2)

        self.obs_data['fper'] *= self.obs_data['cbol']
        self.obs_data['u_fper'] = self.obs_data['fper'] * u_fper_frac

    def _load_obs_table(self):
        """Loads observed burst data from file
        """
        path = os.path.dirname(__file__)
        filename = f'{self.system}.dat'
        filepath = os.path.join(path, '..', 'data', 'obs', self.system, filename)

        print(f'Loading obs table: {os.path.abspath(filepath)}')
        self._obs_table = pd.read_csv(filepath, delim_whitespace=True)
        self._obs_table.set_index('epoch', inplace=True, verify_integrity=True)

        try:
            self._obs_table = self._obs_table.loc[list(self.epochs)]
        except KeyError:
            raise KeyError(f'epoch(s) not found in obs_data table')

    # ===============================================================
    #                      Likelihoods
    # ===============================================================
    def lhood(self, x):
        """Returns log-likelihood for given coordinate

        Returns: flt

        Parameters
        ----------
        x : 1Darray
            sample coordinates (must exactly match `params` length and ordering)
        """
        # ===== Prior likelihood =====
        try:
            lp = self.lnprior(x)
        except ZeroLhood:
            return self._zero_lhood

        # ===== Sample burst variables =====
        y_observer = self.sample(x)

        # ===== Evaluate likelihood against observed data =====
        lh = self.compare(y_observer)
        lhood = lp + lh

        return lhood

    def lnprior(self, x):
        """Return prior log-likelihood of sample

        Returns: flt

        Parameters
        ----------
        x : 1Darray
            sample coordinates (must exactly match `params` length and ordering)
        """
        lower_bounds = self._grid_bounds[:, 0]
        upper_bounds = self._grid_bounds[:, 1]

        inside_bounds = np.logical_and(x > lower_bounds,
                                       x < upper_bounds)
        if False in inside_bounds:
            raise ZeroLhood

        prior_lhood = 0.0
        for i, param in enumerate(self.params):
            prior_lhood += np.log(self._priors[param](x[i]))

        return prior_lhood

    def compare(self, y_observer):
        """Returns log-likelihood for all burst variables against observations

        Returns: float

        Parameters
        ----------
        y_observer : [n_epochs, n_bvars]
            burst variables for all epochs in observer frame
        """
        lh = 0.0

        for i, bvar in enumerate(self.bvars):
            bvar_idx = 2 * i
            u_bvar_idx = bvar_idx + 1

            model = y_observer[:, bvar_idx]
            u_model = y_observer[:, u_bvar_idx]

            lh += self._compare_bvar(model=model, u_model=u_model, bvar=bvar)

        return lh

    def _compare_bvar(self, model, u_model, bvar):
        """Returns log-likelihood of given model values

        Calculates difference between modelled and observed values.
        All provided arrays must be the same length

        Parameters
        ----------
        model : 1darray
            Model values for particular property
        u_model : 1darray
            Corresponding model uncertainties
        bvar : str
            burst property being compared
        """
        obs = self.obs_data[bvar]
        u_obs = self.obs_data[f'u_{bvar}']

        weight = self._weights[bvar]
        inv_sigma2 = 1 / (u_model ** 2 + u_obs ** 2)

        lh = -0.5 * weight * ((model - obs) ** 2 * inv_sigma2
                              + np.log(2 * np.pi / inv_sigma2))

        return lh.sum()

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
        self._get_terms()

        self._get_y_local()
        self._get_y_observer()

        return self._y_observer

    def _get_y_local(self):
        """Calculates model values for given coordinates

        Returns: [n_epochs, n_interp_bvars], [n_epochs, n_analytic_bvars]
        """
        self._get_x_epoch()
        self._interpolate()
        self._get_analytic()

        self._y_local = np.concatenate([self._interp_local, self._analytic_local], axis=1)

    def _get_x_epoch(self):
        """Reshapes sample coordinates into epoch array: [n_epochs, n_interp_params]
        """
        for i in range(self._n_epochs):
            for j, key in enumerate(self._epoch_params):
                if key in self._epoch_unique:
                    key = f'{key}{i+1}'

                self._x_epoch[i, j] = self._x_key[key]

    def _interpolate(self):
        """Interpolates burst properties for N epochs

        Returns: [n_epochs, n_interp_bvars]
        """
        self._interp_local = self._grid_interpolator.interpolate(x=self._x_epoch)

        if True in np.isnan(self._interp_local):
            raise ValueError('Sample is outside of model grid')

    def _get_analytic(self):
        """Calculates analytic burst properties
        """
        mdot = self._x_epoch[:, self._epoch_params.index('mdot')]

        analytic = {'fper': mdot * self._terms['mdot_to_lum'],
                    'fedd': self._terms['lum_edd'],
                    }

        for i, bvar in enumerate(self._analytic_bvars):
            idx = 2 * i
            self._analytic_local[:, idx] = analytic[bvar]
            self._analytic_local[:, idx+1] = analytic[bvar] * self._u_frac[bvar]

    # ===============================================================
    #                      Conversions
    # ===============================================================
    def _get_y_observer(self):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame

        Returns: [n_epochs, n_bvars]
        """
        for i, bvar in enumerate(self.bvars):
            i0 = 2 * i
            i1 = i0 + 2

            values = self._y_local[:, i0:i1]
            self._y_observer[:, i0:i1] = self._shift_to_observer(values=values, bvar=bvar)

    def _shift_to_observer(self, values, bvar):
        """Returns burst property shifted to observer frame/units

        Returns: float or [float]

        Parameters
        ----------
        values : flt or ndarray
            model frame value(s)
        bvar : str
            name of burst property being converted/calculated

        Notes
        ------
        In special case bvar='fper', 'values' must be local accrate
                as fraction of Eddington rate.
        """
        return values * self._terms['shift_factor'][bvar]

    def _get_terms(self):
        """Calculate derived terms
        """
        mass_nw = gravity.mass_from_g(g_nw=self._x_key['g'],
                                      r_nw=self._kepler_radius)

        phi = self._x_key['mass'] / mass_nw

        r_ratio = gravity.get_xi(r_nw=self._kepler_radius,
                                 m_nw=mass_nw,
                                 phi=phi)

        redshift = gravity.redshift_from_xi_phi(phi=phi, xi=r_ratio)

        self._terms['lum_edd'] = accretion.edd_lum_newt(m_nw=mass_nw,
                                                        x=self._x_key['x'])

        potential = -gravity.potential_from_redshift(redshift)
        self._terms['mdot_to_lum'] = self._mdot_edd * potential

        # local to observer conversions
        burst_lum_to_flux = 4 * np.pi * (self._x_key['d_b'] * self._kpc_to_cm)**2
        pers_lum_to_flux = burst_lum_to_flux * self._x_key['xi_ratio']

        fluence_factor = phi / burst_lum_to_flux
        burst_factor = phi / (burst_lum_to_flux * redshift)
        pers_factor = phi / (pers_lum_to_flux * redshift)

        self._terms['shift_factor'] = {'rate': 1 / redshift,
                                       'fluence': fluence_factor,
                                       'peak': burst_factor,
                                       'fedd': burst_factor,
                                       'fper': pers_factor,
                                       }

    # ===============================================================
    #                      Misc.
    # ===============================================================
    def _fill_x_key(self):
        """Fills dictionary of sample coordinates
        """
        for i, key in enumerate(self.params):
            self._x_key[key] = self._x[i]
