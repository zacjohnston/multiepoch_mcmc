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
                 discard=None,
                 thin=None,
                 tau=None,
                 ):
        """
        Parameters
        ----------
        system : str
        n_walkers : int
        discard : bool
        thin : bool
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

        if (discard is None) or (thin is None) or (tau is None):
            print('Calculating autocorrelation time')
            self.tau = self._backend.get_autocorr_time(tol=0)
            self.discard = int(2 * self.tau.max())
            self.thin = int(0.5 * self.tau.min())
        else:
            self.tau = tau
            self.discard = discard
            self.thin = thin

        self.n_autocorr = self.n_steps / self.tau.mean()

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

    def print_summary(self):
        """Print summary statistics for 1D marginilized posteriors
        """
        max_len = len(max(self.params, key=len))

        for param, summ in self.summary.items():
            if None in summ:
                print(f'{param.ljust(max_len)} = unconstrained!')
            else:
                val = summ[1]
                _min = summ[0]
                _max = summ[2]

                minus = val - _min
                plus = _max - val

                print(f'{param.ljust(max_len)} = {val:.3f} +{plus:.3f} -{minus:.3f}')
