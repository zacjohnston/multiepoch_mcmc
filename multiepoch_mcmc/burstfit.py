import os
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const

# pyburst
from multiepoch_mcmc import accretion, gravity


class ZeroLhood(Exception):
    """Escape calculation if likelihood is zero
    """
    pass


class BurstFit:
    """Class for comparing modelled bursts to observed bursts
    """
    _c = const.c.to(u.cm / u.s)                             # speed of light
    _mdot_edd = 1.75e-8 * (u.M_sun / u.year).to(u.g / u.s)  # eddington accretion rate
    _ref_radius = 10                                        # reference NS raius [km]

    def __init__(self,
                 grid_interpolator,
                 priors,
                 grid_bounds,
                 weights,
                 bprops=('rate', 'fluence', 'peak', 'fper', 'fedd'),
                 analytic_bprops=('fper', 'fedd'),
                 interp_bprops=('rate', 'fluence', 'peak'),
                 interp_keys=('mdot', 'x', 'z', 'qb', 'mass'),
                 param_keys=('mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'),
                 epoch_unique=('mdot', 'qb'),
                 system='gs1826',
                 epochs=(1998, 2000, 2007),
                 u_fper_frac=0.0,
                 u_fedd_frac=0.0,
                 zero_lhood=-np.inf):
        """"""
        self.system = system
        self.epochs = epochs
        self._n_epochs = len(self.epochs)

        self.param_keys = param_keys
        self.interp_keys = interp_keys
        self.epoch_unique = epoch_unique
        self._param_aliases = {'mass': 'm_nw'}
        
        self.bprops = bprops
        self.interp_bprops = interp_bprops
        self.analytic_bprops = analytic_bprops
        
        self._grid_bounds = grid_bounds
        self.weights = weights
        self._zero_lhood = zero_lhood
        self._u_fper_frac = u_fper_frac
        self._u_fedd_frac = u_fedd_frac
        self._priors = priors
        self._grid_interpolator = grid_interpolator

        self._obs_table = None
        self.obs_data = None

        self._unpack_obs_data()

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
    #                      Calculation
    # ===============================================================
    def lhood(self, x):
        """Returns log-likelihood for given coordinate

        Returns: flt

        Parameters
        ----------
        x : 1D array
            coordinates of sample point (must match length and ordering of `param_keys`)
        """
        x_dict = self._get_x_dict(x=x)

        # ===== check priors =====
        try:
            lp = self.lnprior(x=x, x_dict=x_dict)
        except ZeroLhood:
            return self._zero_lhood

        # ===== Interpolate and calculate local model burst properties =====
        try:
            interp_local, analytic_local = self._get_model_local(x_dict=x_dict)
        except ZeroLhood:
            return self._zero_lhood

        # ===== Shift all burst properties to observable quantities =====
        interp_shifted, analytic_shifted = self._get_model_shifted(
                                                    interp_local=interp_local,
                                                    analytic_local=analytic_local,
                                                    x_dict=x_dict)

        # ===== Evaluate likelihoods against observed data =====
        lh = self._compare_all(interp_shifted, analytic_shifted)
        lhood = lp + lh

        return lhood

    def bprop_sample(self, x, x_dict=None):
        """Returns the predicted observables for a given sample of parameters

        Effectively performs lhood() without the lhood parts

        parameters
        ----------
        x : np.array
            set of parameters, same as input to lhood()
        x_dict : dict
            pass param dict directly
        """
        if x_dict is None:
            x_dict = self._get_x_dict(x=x)
        interp_local, analytic_local = self._get_model_local(x_dict=x_dict)

        interp_shifted, analytic_shifted = self._get_model_shifted(
                                                    interp_local=interp_local,
                                                    analytic_local=analytic_local,
                                                    x_dict=x_dict)

        return np.concatenate([interp_shifted, analytic_shifted], axis=1)

    def _get_x_dict(self, x):
        """Returns coordinates as dictionary

        Returns: {param: value}

        Parameters
        ----------
        x : 1Darray
            coordinates of sample
        """
        x_dict = {}
        for i, key in enumerate(self.param_keys):
            x_dict[key] = x[i]

        return x_dict

    def _get_model_local(self, x_dict):
        """Calculates model values for given coordinates
            Returns: interp_local, analytic_local
        """
        epoch_params = self._get_epoch_params(x_dict=x_dict)
        interp_local = self._get_interp_bprops(interp_params=epoch_params)
        analytic_local = self._get_analytic_bprops(x_dict=x_dict,
                                                   epoch_params=epoch_params)

        return interp_local, analytic_local

    def _get_analytic_bprops(self, x_dict, epoch_params):
        """Returns calculated analytic burst properties for given x_dict
        """
        def get_fedd():
            """Returns Eddington flux array (n_epochs, 2)
                Note: Actually the luminosity at this stage, as this is the local value
            """
            out = np.full([self._n_epochs, 2], np.nan, dtype=float)
            x_edd = x_dict['x']

            l_edd = accretion.edd_lum_newt(mass=x_dict['m_nw'], x=x_edd)
            out[:, 0] = l_edd
            out[:, 1] = l_edd * self._u_fedd_frac
            return out

        def get_fper():
            """Returns persistent accretion flux array (n_epochs, 2)
                Note: Actually the luminosity, because this is the local value
            """
            out = np.full([self._n_epochs, 2], np.nan, dtype=float)
            mass_ratio, redshift = self._get_gr_factors(x_dict=x_dict)

            phi = (redshift - 1) * self._c.value ** 2 / redshift  # grav potential
            mdot = epoch_params[:, self.interp_keys.index('mdot')]
            l_per = mdot * self._mdot_edd * phi

            out[:, 0] = l_per
            out[:, 1] = out[:, 0] * self._u_fper_frac
            return out

        function_map = {'fper': get_fper, 'fedd': get_fedd}
        analytic = np.full([self._n_epochs, 2*len(self.analytic_bprops)],
                           np.nan,
                           dtype=float)

        for i, bprop in enumerate(self.analytic_bprops):
            analytic[:, 2*i: 2*(i+1)] = function_map[bprop]()

        return analytic

    def _get_model_shifted(self, interp_local, analytic_local, x_dict):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame
        """
        interp_shifted = np.full_like(interp_local, np.nan, dtype=float)
        analytic_shifted = np.full_like(analytic_local, np.nan, dtype=float)

        # ==== shift interpolated bprops ====
        # TODO: concatenate bprop arrays and handle together
        for i, bprop in enumerate(self.interp_bprops):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            interp_shifted[:, i0:i1] = self._shift_to_observer(
                                                    values=interp_local[:, i0:i1],
                                                    bprop=bprop,
                                                    x_dict=x_dict)

        # ==== shift analytic bprops ====
        for i, bprop in enumerate(self.analytic_bprops):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            analytic_shifted[:, i0:i1] = self._shift_to_observer(
                                                    values=analytic_local[:, i0:i1],
                                                    bprop=bprop,
                                                    x_dict=x_dict)
        return interp_shifted, analytic_shifted

    def _compare_all(self, interp_shifted, analytic_shifted):
        """Compares all bprops against observations and returns total likelihood
        """
        lh = 0.0
        all_shifted = np.concatenate([interp_shifted, analytic_shifted], axis=1)

        for i, bprop in enumerate(self.bprops):
            bprop_idx = 2 * i
            u_bprop_idx = bprop_idx + 1

            model = all_shifted[:, bprop_idx]
            u_model = all_shifted[:, u_bprop_idx]

            lh += self.compare(model=model, u_model=u_model, bprop=bprop)

        return lh

    def _shift_to_observer(self, values, bprop, x_dict):
        """Returns burst property shifted to observer frame/units

        Parameters
        ----------
        values : flt or ndarray
            model frame value(s)
        bprop : str
            name of burst property being converted/calculated
        x_dict : 1darray
            parameters (see param_keys)

        Notes
        ------
        In special case bprop='fper', 'values' must be local accrate
                as fraction of Eddington rate.
        """
        kpc_to_cm = u.kpc.to(u.cm)
        mass_ratio, redshift = self._get_gr_factors(x_dict=x_dict)

        flux_factor_b = 4 * np.pi * (x_dict['d_b'] * kpc_to_cm) ** 2
        flux_factor_p = flux_factor_b * x_dict['xi_ratio']

        flux_factors = {'dt': 1,
                        'rate': 1,
                        'fluence': flux_factor_b,
                        'peak': flux_factor_b,
                        'fedd': flux_factor_b,
                        'fper': flux_factor_p,
                        'tail_index': 1,
                        }

        gr_corrections = {'dt': redshift / 3600,  # include hr to sec
                          'rate': 1/redshift,
                          'fluence': mass_ratio,
                          'peak': mass_ratio / redshift,
                          'fedd': mass_ratio / redshift,
                          'fper': mass_ratio / redshift,
                          'tail_index': 1,
                          }

        flux_factor = flux_factors.get(bprop)
        gr_correction = gr_corrections.get(bprop)

        if flux_factor is None:
            raise ValueError('bprop must be one of (dt, rate, fluence, peak, '
                             'fper, f_edd, tail_index)')

        shifted = (values * gr_correction) / flux_factor

        return shifted

    def _get_interp_bprops(self, interp_params):
        """Interpolates burst properties for N epochs

        Parameters
        ----------
        interp_params : 1darray
            parameters specific to the model (e.g. mdot1, x, z, qb, get_mass)
        """
        output = self._grid_interpolator.interpolate(x=interp_params)

        if True in np.isnan(output):
            raise ZeroLhood

        return output

    def _get_epoch_params(self, x_dict):
        """Extracts array of model parameters for each epoch
        """
        epoch_params = np.full((self._n_epochs, len(self.interp_keys)),
                               np.nan,
                               dtype=float)

        for i in range(self._n_epochs):
            for j, key in enumerate(self.interp_keys):
                epoch_params[i, j] = self._get_interp_param(key=key,
                                                            x_dict=x_dict,
                                                            epoch_idx=i)

        return epoch_params

    def _get_interp_param(self, key, x_dict, epoch_idx):
        """Extracts interp param value from full x_dict
        """
        key = self._param_aliases.get(key, key)

        if key in self.epoch_unique:
            key = f'{key}{epoch_idx + 1}'

        return x_dict[key]

    def _get_gr_factors(self, x_dict):
        """Returns GR factors (m_ratio, redshift) given (m_nw, m_gr)"""
        mass_nw = x_dict['m_nw']
        mass_gr = x_dict['m_gr']
        m_ratio = mass_gr / mass_nw
        _, redshift = gravity.gr_corrections(r=self._ref_radius,
                                             m=mass_nw,
                                             phi=m_ratio)

        return m_ratio, redshift

    def lnprior(self, x, x_dict):
        """Return log-likelihood of prior
        """
        lower_bounds = self._grid_bounds[:, 0]
        upper_bounds = self._grid_bounds[:, 1]

        inside_bounds = np.logical_and(x > lower_bounds,
                                       x < upper_bounds)
        if False in inside_bounds:
            raise ZeroLhood

        prior_lhood = 0.0
        for key, val in x_dict.items():
            prior_lhood += np.log(self._priors[key](val))

        return prior_lhood

    def compare(self, model, u_model, bprop):
        """Returns log-likelihood of given model values

        Calculates difference between modelled and observed values.
        All provided arrays must be the same length

        Parameters
        ----------
        model : 1darray
            Model values for particular property
        u_model : 1darray
            Corresponding model uncertainties
        bprop : str
            burst property being compared
        """
        obs = self.obs_data[bprop]
        u_obs = self.obs_data[f'u_{bprop}']

        weight = self.weights[bprop]
        inv_sigma2 = 1 / (u_model ** 2 + u_obs ** 2)
        lh = -0.5 * weight * ((model - obs) ** 2 * inv_sigma2
                              + np.log(2 * np.pi / inv_sigma2))

        return lh.sum()
