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
    _c = const.c.to(u.cm / u.s).value                       # speed of light
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
        self._interp_params = self._config['keys']['interp_params']
        self._epoch_params = self._config['keys']['epoch_params']
        self.n_dim = len(self.params)

        self.bvars = self._config['keys']['bvars']
        self._interp_bvars = self._config['keys']['interp_bvars']
        self._analytic_bvars = self._config['keys']['analytic_bvars']
        
        self._grid_bounds = self._config['grid']['bounds']
        self.x_start = self._config['grid']['x_start']
        self._kepler_radius = self._config['grid']['kepler_radius']

        self._weights = self._config['lhood']['weights']
        self._u_fper_frac = self._config['lhood']['u_fper_frac']
        self._u_fedd_frac = self._config['lhood']['u_fedd_frac']
        self._priors = self._config['lhood']['priors']
        self._obs_table = None
        self.obs_data = None

        # dynamic variables
        self._x_dict = {}
        self._terms = {}
        self._flux_factors = {}

        self._unpack_obs_data()

        self._grid_interpolator = GridInterpolator(file=self._config['interp']['file'],
                                                   params=self._interp_params,
                                                   bvars=self._config['interp']['bvars'])

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
    #                      Likelihood Sampling
    # ===============================================================
    def lhood(self, x):
        """Returns log-likelihood for given coordinate

        Returns: flt

        Parameters
        ----------
        x : 1Darray
            sample coordinates (must exactly match `params` length and ordering)
        """
        self._get_x_dict(x=x)

        # ===== Get prior likelihoods =====
        try:
            lp = self.lnprior(x=x)
        except ZeroLhood:
            return self._zero_lhood

        try:
            self._get_terms()
        except ZeroLhood:
            return self._zero_lhood

        # ===== Interpolate + calculate local burst properties =====
        try:
            interp_local, analytic_local = self._get_model_local()
        except ZeroLhood:
            return self._zero_lhood

        # ===== Shift to observable quantities =====
        y_shifted = self._get_y_shifted(interp_local=interp_local,
                                        analytic_local=analytic_local)

        # ===== Evaluate likelihood against observed data =====
        lh = self.compare(y_shifted)
        lhood = lp + lh

        return lhood

    def lnprior(self, x):
        """Return log-likelihood of prior

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
        for key, val in self._x_dict.items():
            prior_lhood += np.log(self._priors[key](val))

        return prior_lhood

    def compare(self, y_shifted):
        """Returns log-likelihood for all burst variables against observations

        Returns: float

        Parameters
        ----------
        y_shifted : [n_epochs, n_bvars]
            burst variables for all epochs in observer frame
        """
        lh = 0.0

        for i, bvar in enumerate(self.bvars):
            bvar_idx = 2 * i
            u_bvar_idx = bvar_idx + 1

            model = y_shifted[:, bvar_idx]
            u_model = y_shifted[:, u_bvar_idx]

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

    def sample(self, x):
        """Returns the predicted observables for given coordinates

        Effectively performs lhood() without the likelihood calculations

        Returns: [n_epochs, n_bvars]

        Parameters
        ----------
        x : 1Darray
            sample coordinates
        """
        self._get_x_dict(x=x)
        self._get_terms()

        interp_local, analytic_local = self._get_model_local()

        y_shifted = self._get_y_shifted(interp_local=interp_local,
                                        analytic_local=analytic_local)

        return y_shifted

    # ===============================================================
    #                      Burst Sampling
    # ===============================================================
    def _get_model_local(self):
        """Calculates model values for given coordinates

        Returns: [n_epochs, n_interp_bvars], [n_epochs, n_analytic_bvars]
        """
        x_interp = self._get_x_interp()
        interp_local = self._get_interp_bvars(x_interp=x_interp)
        analytic_local = self._get_analytic_bvars(x_interp=x_interp)

        return interp_local, analytic_local

    def _get_interp_bvars(self, x_interp):
        """Interpolates burst properties for N epochs

        Returns: [n_epochs, n_interp_bvars]

        Parameters
        ----------
        x_interp : 1darray
            parameters specific to the model (e.g. mdot1, x, z, qb, get_mass)
        """
        output = self._grid_interpolator.interpolate(x=x_interp)

        if True in np.isnan(output):
            raise ZeroLhood

        return output

    def _get_analytic_bvars(self, x_interp):
        """Returns calculated analytic burst properties

        Returns: [n_epochs, n_analytic_bvars]

        Parameters
        ----------
        x_interp : [n_epochs, n_analytic_params]
        """
        function_map = {'fper': self._get_fper, 'fedd': self._get_fedd}
        analytic = np.full([self._n_epochs, 2*len(self._analytic_bvars)],
                           np.nan,
                           dtype=float)

        for i, bvar in enumerate(self._analytic_bvars):
            analytic[:, 2*i: 2*(i+1)] = function_map[bvar](x_interp)

        return analytic

    def _get_fedd(self, x_interp):
        """Returns Eddington flux array (n_epochs, 2)
            Note: Actually luminosity, as this is the local value

        Returns: [n_epochs, float]
        """
        out = np.full([self._n_epochs, 2], np.nan, dtype=float)

        l_edd = accretion.edd_lum_newt(m_nw=self._terms['mass_nw'],
                                       x=self._x_dict['x'])

        out[:, 0] = l_edd
        out[:, 1] = l_edd * self._u_fedd_frac

        return out

    def _get_fper(self, x_interp):
        """Returns persistent accretion flux array (n_epochs, 2)
            Note: Actually luminosity, as this is the local value

        Returns: [n_epochs, float]

        Parameters
        ----------
        x_interp : [n_epochs, n_interp_params]
        """
        out = np.full([self._n_epochs, 2], np.nan, dtype=float)

        mdot = x_interp[:, self._interp_params.index('mdot')]
        l_per = mdot * self._mdot_edd * self._terms['potential']

        out[:, 0] = l_per
        out[:, 1] = out[:, 0] * self._u_fper_frac
        return out

    def _get_x_interp(self):
        """Returns epoch array of coordinates

        Returns: [n_epochs, n_interp_params]
        """
        x_interp = np.full((self._n_epochs, len(self._interp_params)),
                           np.nan,
                           dtype=float)

        for i in range(self._n_epochs):
            for j, key in enumerate(self._interp_params):
                x_interp[i, j] = self._get_interp_param(key=key,
                                                        epoch_idx=i)

        return x_interp

    def _get_interp_param(self, key, epoch_idx):
        """Extracts interp param value from full x_dict

        Returns: float

        Parameters
        ----------
        key : str
        epoch_idx : int
        """
        if key in self._epoch_params:
            key = f'{key}{epoch_idx + 1}'

        return self._x_dict[key]

    # ===============================================================
    #                      Conversions
    # ===============================================================
    def _get_y_shifted(self, interp_local, analytic_local):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame

        Returns: [n_epochs, n_bvars]

        Parameters
        ----------
        interp_local: [n_epochs, n_interp_bvars]
        analytic_local: [n_epochs, n_analytic_bvars]
        """
        interp_shifted = np.full_like(interp_local, np.nan, dtype=float)
        analytic_shifted = np.full_like(analytic_local, np.nan, dtype=float)

        # ==== shift interpolated bvars ====
        # TODO: concatenate bvar arrays and handle together
        for i, bvar in enumerate(self._interp_bvars):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            interp_shifted[:, i0:i1] = self._shift_to_observer(
                values=interp_local[:, i0:i1],
                bvar=bvar)

        # ==== shift analytic bvars ====
        for i, bvar in enumerate(self._analytic_bvars):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            analytic_shifted[:, i0:i1] = self._shift_to_observer(
                values=analytic_local[:, i0:i1],
                bvar=bvar)

        y_shifted = np.concatenate([interp_shifted, analytic_shifted], axis=1)

        return y_shifted

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
        gr_factor = self._terms['gr_factor'][bvar]
        flux_factor = self._terms['flux_factor'][bvar]

        shifted = (values * gr_factor) / flux_factor

        return shifted

    def _get_terms(self):
        """Calculate derived terms
        """
        self._terms['mass_nw'] = gravity.mass_from_g(g_nw=self._x_dict['g'],
                                                     r_nw=self._kepler_radius)

        self._terms['mass_ratio'] = self._x_dict['mass'] / self._terms['mass_nw']

        self._terms['r_ratio'] = gravity.get_xi(r_nw=self._kepler_radius,
                                                m_nw=self._terms['mass_nw'],
                                                phi=self._terms['mass_ratio'])

        self._terms['redshift'] = gravity.redshift_from_xi_phi(
                                                phi=self._terms['mass_ratio'],
                                                xi=self._terms['r_ratio'])

        self._terms['potential'] = -gravity.potential_from_redshift(self._terms['redshift'])

        self._flux_factors['burst'] = 4 * np.pi * (self._x_dict['d_b'] * self._kpc_to_cm)**2
        self._flux_factors['pers'] = self._flux_factors['burst'] * self._x_dict['xi_ratio']

        self._terms['flux_factor'] = {'rate': 1,
                                      'fluence': self._flux_factors['burst'],
                                      'peak': self._flux_factors['burst'],
                                      'fedd': self._flux_factors['burst'],
                                      'fper': self._flux_factors['pers'],
                                      }

        self._terms['gr_factor'] = {'rate': 1 / self._terms['redshift'],
                                    'fluence': self._terms['mass_ratio'],
                                    'peak': self._terms['mass_ratio'] / self._terms['redshift'],
                                    'fedd': self._terms['mass_ratio'] / self._terms['redshift'],
                                    'fper': self._terms['mass_ratio'] / self._terms['redshift'],
                                    }

    # ===============================================================
    #                      Misc.
    # ===============================================================
    def _get_x_dict(self, x):
        """Returns sample coordinates as dictionary

        Returns: {param: value}

        Parameters
        ----------
        x : 1Darray
            coordinates of sample
        """
        for i, key in enumerate(self.params):
            self._x_dict[key] = x[i]
