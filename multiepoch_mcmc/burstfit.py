import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

# pyburst
from multiepoch_mcmc import gravity

c = const.c.to(u.cm / u.s)
msunyr_to_gsec = (u.M_sun / u.year).to(u.g / u.s)
mdot_edd = 1.75e-8 * msunyr_to_gsec
z_sun = 0.01


class ZeroLhood(Exception):
    pass


class BurstFit:
    """Class for comparing modelled bursts to observed bursts
    """

    def __init__(self,
                 source,
                 verbose=True,
                 lhood_factor=1,
                 debug=False,
                 priors_only=False,
                 re_interp=False,
                 u_fper_frac=0.0,
                 u_fedd_frac=0.0,
                 zero_lhood=-np.inf,
                 reference_radius=10):
        """
        reference_radius : float
            Newtonian get_radius (km) used in Kepler
        """
        self.source = source
        self.verbose = verbose

        self.param_idxs = {}
        self.interp_idxs = {}
        self.get_indexes()

        self.reference_radius = reference_radius

        self.n_bprops = len(self.mcmc_version.bprops)
        self.n_analytic_bprops = len(self.mcmc_version.analytic_bprops)
        self.n_interp_params = len(self.mcmc_version.interp_keys)
        self.has_xedd_ratio = ('xedd_ratio' in self.mcmc_version.param_keys)
        self.constants = self.mcmc_version.constants

        self.kpc_to_cm = u.kpc.to(u.cm)
        self.zero_lhood = zero_lhood
        self.u_fper_frac = u_fper_frac
        self.u_fedd_frac = u_fedd_frac
        self.lhood_factor = lhood_factor
        self.priors_only = priors_only

        if self.mcmc_version.synthetic:
            interp_source = self.mcmc_version.interp_source
        else:
            interp_source = self.source

        self.kemulator = interpolator.Kemulator(source=interp_source,
                                                version=self.mcmc_version.interpolator,
                                                re_interp=re_interp)
        self.obs = None
        self.n_epochs = None
        self.obs_data = None
        self.extract_obs_values()

        self.priors = self.mcmc_version.priors

    def printv(self, string, **kwargs):
        if self.verbose:
            print(string, **kwargs)

    def get_indexes(self):
        """Extracts indexes of parameters and burst properties

        Expects params array to be in same order as param_keys
        """
        def idx_dict(dict_in):
            dict_out = {}
            for i, key in enumerate(dict_in):
                dict_out[key] = i
            return dict_out

        self.param_idxs = idx_dict(self.mcmc_version.param_keys)
        self.interp_idxs = idx_dict(self.mcmc_version.interp_keys)

    def extract_obs_values(self):
        """Unpacks observed burst properties (dt, fper, etc.) from data
        """
        if self.mcmc_version.synthetic:
            self.obs_data = synth.extract_obs_data(self.source,
                                                   self.mcmc_version.synth_version,
                                                   group=self.mcmc_version.synth_group)
            self.n_epochs = len(self.obs_data['fluence'])
        else:
            filename = f'{self.source_obs}.dat'
            filepath = os.path.join(PYBURST_PATH, 'files', 'obs_data',
                                    self.source_obs, filename)

            self.obs = pd.read_csv(filepath, delim_whitespace=True)
            self.obs.set_index('epoch', inplace=True, verify_integrity=True)

            # Select single epoch (if applicable)
            if self.mcmc_version.epoch is not None:
                # TODO: define/specify epochs for all mcmc versions?
                try:
                    self.obs = self.obs.loc[[self.mcmc_version.epoch]]
                except KeyError:
                    raise KeyError(f'epoch [{self.mcmc_version.epoch}] '
                                   f'not in obs_data table')

            self.n_epochs = len(self.obs)
            self.obs_data = self.obs.to_dict(orient='list')

            for key, item in self.obs_data.items():
                self.obs_data[key] = np.array(item)

            # ===== Apply bolometric corrections (cbol) to fper ======
            u_fper_frac = np.sqrt((self.obs_data['u_cbol'] / self.obs_data['cbol']) ** 2
                                  + (self.obs_data['u_fper'] / self.obs_data['fper']) ** 2)

            self.obs_data['fper'] *= self.obs_data['cbol']
            self.obs_data['u_fper'] = self.obs_data['fper'] * u_fper_frac

    def lhood(self, x, plot=False):
        """Return lhood for given params

        Parameters
        ----------
        x : 1D array
            set of parameter values to try (must match order of mcmc_version.param_keys)
        plot : bool
            whether to plot the comparison
        """
        params = self.get_params_dict(x=x)
        zero_lhood = self.zero_lhood * self.lhood_factor

        # ===== check priors =====
        try:
            lp = self.lnprior(x=x, params=params)
        except ZeroLhood:
            return zero_lhood

        if self.priors_only:
            return lp * self.lhood_factor

        # ===== Interpolate and calculate local model burst properties =====
        try:
            interp_local, analytic_local = self.get_model_local(params=params)
        except ZeroLhood:
            return zero_lhood

        # ===== Shift all burst properties to observable quantities =====
        interp_shifted, analytic_shifted = self.get_model_shifted(
                                                    interp_local=interp_local,
                                                    analytic_local=analytic_local,
                                                    params=params)

        # ===== Setup plotting =====
        n_bprops = len(self.mcmc_version.bprops)
        if plot:
            n_rows = int(np.ceil(n_bprops/2))
            subplot_width = 3
            subplot_height = 2.5

            fig, ax = plt.subplots(n_rows, 2, sharex=True,
                                   figsize=(2*subplot_width, n_rows * subplot_height))
            if n_bprops % 2 == 1:
                ax[-1, -1].axis('off')
        else:
            fig = ax = None

        # ===== Evaluate likelihoods against observed data =====
        lh = self.compare_all(interp_shifted, analytic_shifted, ax=ax, plot=plot)
        lhood = (lp + lh) * self.lhood_factor

        # ===== Finalise plotting =====
        if plot:
            plt.show(block=False)
            return lhood, fig
        else:
            return lhood

    def bprop_sample(self, x, params=None):
        """Returns the predicted observables for a given sample of parameters

        Effectively performs lhood() without the lhood parts

        parameters
        ----------
        x : np.array
            set of parameters, same as input to lhood()
        params : dict
            pass param dict directly
        """
        if params is None:
            params = self.get_params_dict(x=x)
        interp_local, analytic_local = self.get_model_local(params=params)

        interp_shifted, analytic_shifted = self.get_model_shifted(
                                                    interp_local=interp_local,
                                                    analytic_local=analytic_local,
                                                    params=params)

        return np.concatenate([interp_shifted, analytic_shifted], axis=1)

    def get_params_dict(self, x):
        """Returns params in form of dict
        """
        keys = self.mcmc_version.param_keys
        params_dict = dict.fromkeys(keys)

        for i, key in enumerate(keys):
            params_dict[key] = x[i]

        for key, val in self.constants.items():
            params_dict[key] = val

        return params_dict

    def get_model_local(self, params):
        """Calculates predicted model values (bprops) for given params
            Returns: interp_local, analytic_local
        """
        epoch_params = self.get_epoch_params(params=params)
        interp_local = self.get_interp_bprops(interp_params=epoch_params)
        analytic_local = self.get_analytic_bprops(params=params, epoch_params=epoch_params)

        return interp_local, analytic_local

    def get_analytic_bprops(self, params, epoch_params):
        """Returns calculated analytic burst properties for given params
        """
        def get_fedd():
            """Returns Eddington flux array (n_epochs, 2)
                Note: Actually the luminosity at this stage, as this is the local value
            """
            out = np.full([self.n_epochs, 2], np.nan, dtype=float)

            if self.has_xedd_ratio:
                x_edd = params['x'] * params['xedd_ratio']
            elif self.mcmc_version.x_edd_option == 'x_0':
                x_edd = params['x']
            else:
                x_edd = self.mcmc_version.x_edd_option

            l_edd = accretion.eddington_lum_newtonian(mass=params['m_nw'], x=x_edd)
            out[:, 0] = l_edd
            out[:, 1] = l_edd * self.u_fedd_frac
            return out

        def get_fper():
            """Returns persistent accretion flux array (n_epochs, 2)
                Note: Actually the luminosity, because this is the local value
            """
            out = np.full([self.n_epochs, 2], np.nan, dtype=float)
            mass_ratio, redshift = self.get_gr_factors(params=params)

            phi = (redshift - 1) * c.value ** 2 / redshift  # gravitational potential
            mdot = epoch_params[:, self.interp_idxs['mdot']]
            l_per = mdot * mdot_edd * phi

            out[:, 0] = l_per
            out[:, 1] = out[:, 0] * self.u_fper_frac
            return out

        function_map = {'fper': get_fper, 'fedd': get_fedd}
        analytic = np.full([self.n_epochs, 2*self.n_analytic_bprops], np.nan, dtype=float)

        for i, bprop in enumerate(self.mcmc_version.analytic_bprops):
            analytic[:, 2*i: 2*(i+1)] = function_map[bprop]()

        return analytic

    def get_model_shifted(self, interp_local, analytic_local, params):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame
        """
        interp_shifted = np.full_like(interp_local, np.nan, dtype=float)
        analytic_shifted = np.full_like(analytic_local, np.nan, dtype=float)

        # ==== shift interpolated bprops ====
        # TODO: concatenate bprop arrays and handle together
        for i, bprop in enumerate(self.mcmc_version.interp_bprops):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            interp_shifted[:, i0:i1] = self.shift_to_observer(
                                                    values=interp_local[:, i0:i1],
                                                    bprop=bprop, params=params)

        # ==== shift analytic bprops ====
        for i, bprop in enumerate(self.mcmc_version.analytic_bprops):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            analytic_shifted[:, i0:i1] = self.shift_to_observer(
                                                    values=analytic_local[:, i0:i1],
                                                    bprop=bprop, params=params)
        return interp_shifted, analytic_shifted

    def compare_all(self, interp_shifted, analytic_shifted, ax, plot=False):
        """Compares all bprops against observations and returns total likelihood
        """
        lh = 0.0
        all_shifted = np.concatenate([interp_shifted, analytic_shifted], axis=1)

        for i, bprop in enumerate(self.mcmc_version.bprops):
            bprop_idx = 2 * i
            u_bprop_idx = bprop_idx + 1

            model = all_shifted[:, bprop_idx]
            u_model = all_shifted[:, u_bprop_idx]

            lh += self.compare(model=model, u_model=u_model, bprop=bprop, label=bprop)

        return lh

    def shift_to_observer(self, values, bprop, params):
        """Returns burst property shifted to observer frame/units

        Parameters
        ----------
        values : ndarray|flt
            model frame value(s)
        bprop : str
            name of burst property being converted/calculated
        params : 1darray
            parameters (see param_keys)


        Notes
        ------
        In special case bprop='fper', 'values' must be local accrate
                as fraction of Eddington rate.
        """
        mass_ratio, redshift = self.get_gr_factors(params=params)
        flux_factor_b = 4 * np.pi * (self.kpc_to_cm * params['d_b']) ** 2
        flux_factor_p = flux_factor_b * params['xi_ratio']

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

    def get_interp_bprops(self, interp_params):
        """Interpolates burst properties for N epochs

        Parameters
        ----------
        interp_params : 1darray
            parameters specific to the model (e.g. mdot1, x, z, qb, get_mass)
        """
        output = self.kemulator.emulate_burst(params=interp_params)

        if True in np.isnan(output):
            raise ZeroLhood

        return output

    def get_epoch_params(self, params):
        """Extracts array of model parameters for each epoch
        """
        epoch_params = np.full((self.n_epochs, self.n_interp_params), np.nan, dtype=float)

        for i in range(self.n_epochs):
            for j, key in enumerate(self.mcmc_version.interp_keys):
                epoch_params[i, j] = self.get_interp_param(key, params, epoch_idx=i)

        return epoch_params

    def get_interp_param(self, key, params, epoch_idx):
        """Extracts interp param value from full params
        """
        key = self.mcmc_version.param_aliases.get(key, key)

        if key in self.mcmc_version.epoch_unique:
            key = f'{key}{epoch_idx + 1}'

        return params[key]

    def get_gr_factors(self, params):
        """Returns GR factors (m_ratio, redshift) given (m_nw, m_gr)"""
        mass_nw = params['m_nw']
        mass_gr = params['m_gr']
        m_ratio = mass_gr / mass_nw
        redshift = gravity.gr_corrections(r=self.reference_radius, m=mass_nw, phi=m_ratio)[1]
        return m_ratio, redshift

    def lnprior(self, x, params):
        """Return logarithm prior lhood of params
        """
        lower_bounds = self.mcmc_version.grid_bounds[:, 0]
        upper_bounds = self.mcmc_version.grid_bounds[:, 1]
        inside_bounds = np.logical_and(x > lower_bounds,
                                       x < upper_bounds)

        if False in inside_bounds:
            raise ZeroLhood

        prior_lhood = 0.0
        for key, val in params.items():
            if key not in self.constants:
                prior_lhood += np.log(self.priors[key](val))

        return prior_lhood

    def compare(self, model, u_model, bprop, label='', plot=False):
        """Returns logarithmic likelihood of given model values

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
        label : str
            label of parameter to print
        plot : bool
            whether to plot the comparison
        """
        obs = self.obs_data[bprop]
        u_obs = self.obs_data[f'u_{bprop}']

        pyprint.check_same_length(model, obs, 'model and obs arrays')
        pyprint.check_same_length(u_model, u_obs, 'u_model and u_obs arrays')

        weight = self.mcmc_version.weights[bprop]
        inv_sigma2 = 1 / (u_model ** 2 + u_obs ** 2)
        lh = -0.5 * weight * ((model - obs) ** 2 * inv_sigma2
                              + np.log(2 * np.pi / inv_sigma2))

        if plot:
            self.plot_compare(model=model, u_model=u_model, bprop=label)
        return lh.sum()
