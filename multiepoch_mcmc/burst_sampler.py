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
    model : BurstModel
        burst model to sample
    n_dim : int
        number of model dimensions (params)
    obs : ObsData
        observed burst data to compare to
    params : [str]
        model parameter keys
    system : str
        name of bursting system being modelled
    weights : [n_bvars]
        weight assigned to each bvar when computing likelihood
    x_start : [n_dim]
        coordinates of starting point for sampling

    Methods
    -------
    lhood(x)
        Calculates log-likelihood for given sample coordinates
    compare(model, u_model, bvar)
        Returns log-likelihood for given model-observation comparison
    sample(x)
        Returns the modelled burst variables at given sample coordinates
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
        self._config = config.load_system_config(system=self.system)
        self.epochs = self._config['obs']['epochs']

        self.params = self._config['keys']['params']
        self.bvars = self._config['keys']['bvars']
        self.n_dim = len(self.params)

        self._grid_bounds = self._config['grid']['bounds']
        self._priors = self._config['lhood']['priors']
        self.x_start = self._config['grid']['x_start']

        self.weights = np.zeros(len(self.bvars))
        for i, bvar in enumerate(self.bvars):
            self.weights[i] = self._config['lhood']['weights'][bvar]

        self.obs = ObsData(system=self.system, epochs=self.epochs)
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

        # ===== Sample model burst variables =====
        y, u_y = self.model.sample(x)

        # ===== Evaluate likelihood against observed data =====
        lh = self.compare(y, u_y)
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

    def compare(self, y, u_y):
        """Returns log-likelihood for model burst(s) against observations

        Returns: float

        Parameters
        ----------
        y : [n_epochs, n_bvars]
            model burst variables in observer frame
        u_y : [n_epochs, n_bvars]
            corresponding uncertainties
        """
        inv_sigma2 = 1 / (u_y**2 + self.obs.u_y**2)

        lh = -0.5 * self.weights * (inv_sigma2 * (y - self.obs.y)**2
                                    + np.log(2 * np.pi / inv_sigma2))

        return lh.sum()
