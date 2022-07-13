import numpy as np
import matplotlib.pyplot as plt
import chainconsumer
import emcee

from multiepoch_mcmc import mcmc, config


class MCPlotter:
    """
    Class for plotting MCMC chains
    """

    def __init__(self,
                 system='gs1826',
                 n_walkers=1024,
                 ):
        """
        Parameters
        ----------
        system : str
        n_walkers
        """
        self.system = system
        self.n_walkers = n_walkers

        self._config = config.load_config(system)
        self.params = self._config['keys']['params']
        self.n_dim = len(self.params)

        self._backend = mcmc.open_backend(system=system, n_walkers=n_walkers)

        self.n_steps = self._backend.iteration
        self.filename = self._backend.filename
        self.lhood = self._backend.get_log_prob()
        self.accept_frac = self._backend.accepted.mean() / self.n_steps

        print('Calculating autocorrelation time')
        self.tau = self._backend.get_autocorr_time(tol=0)
        self.thin = int(0.5 * self.tau.min())
        self.discard = int(2 * self.tau.max())

        print('Unpacking chain')
        self.chain = self._backend.get_chain(flat=True,
                                             discard=self.discard,
                                             thin=self.thin)

        self._cc = chainconsumer.ChainConsumer()
        self._cc.add_chain(self.chain, parameters=self.params)

        self._cc.configure(kde=False,
                           smooth=0,
                           sigmas=np.linspace(0, 2, 5),
                           summary=False,
                           usetex=False)

        self.summary = self._cc.get_summary()

    def plot_1d(self,
                filename=None):
        """Plot 1D marginilized posterior distributions

        Parameters
        ----------
        filename : str
        """
        self._cc.plotter.plot_distributions(filename=filename)

    def plot_2d(self,
                filename=None):
        """Plot 2D marginilized posterior distributions (i.e. corner plot)

        Parameters
        ----------
        filename : str
        """
        self._cc.plotter.plot(filename=filename)
