import numpy as np
import os
import matplotlib.pyplot as plt
import chainconsumer

from multiepoch_mcmc import mcmc, config


class MCPlotter:
    """
    Class for plotting MCMC chains
    """

    def __init__(self,
                 system='gs1826',
                 n_walkers=1024,
                 discard=1000,
                 ):
        """
        Parameters
        ----------
        system : str
        n_walkers
        discard : int
        """
        self.system = system
        self.n_walkers = n_walkers
        self.discard = discard

        self._config = config.load_config(system)
        self.params = self._config['keys']['params']
        self.n_dim = len(self.params)

        self._backend = mcmc.open_backend(system=system, n_walkers=n_walkers)

        self.n_steps = self._backend.iteration
        self.chain = self._backend.get_chain(discard=discard)
        self.filename = self._backend.filename
        self.lhood = self._backend.get_log_prob()

        self.accept_frac = self._backend.accepted.mean() / self.n_steps

        self.tau = None
        self.get_autocorr_time()

    def get_autocorr_time(self):
        """Estimate autocorrelation time
        """
        print('Calculating autocorrelation time')
        self.tau = self._backend.get_autocorr_time(discard=self.discard, tol=0)
