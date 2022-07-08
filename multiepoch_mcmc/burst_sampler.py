import numpy as np

from multiepoch_mcmc import config
from multiepoch_mcmc.burst_model import BurstModel
from multiepoch_mcmc.obs_data import ObsData


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
        self._epoch_params = self._config['interp']['params']
        self._n_epochs = len(self.epochs)

        self.params = self._config['keys']['params']
        self.bvars = self._config['keys']['bvars']
        self.n_dim = len(self.params)

        self._grid_bounds = self._config['grid']['bounds']
        self._weights = self._config['lhood']['weights']
        self._priors = self._config['lhood']['priors']
        self.x_start = self._config['grid']['x_start']

        self._obs = ObsData(system=self.system, epochs=self.epochs)
        self.obs_data = self._obs.data

        self.model = BurstModel(system=self.system)

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
        y_observer = self.model.sample(x)

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
