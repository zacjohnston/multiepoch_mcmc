import numpy as np
import matplotlib.pyplot as plt
import chainconsumer

from multiepoch_mcmc import mcmc, config, gravity, anisotropy


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
                 anisotropy_model='he16_a',
                 ):
        """
        Parameters
        ----------
        system : str
        n_walkers : int
        discard : bool
        thin : bool
        anisotropy_model : str
        """
        self.system = system
        self.n_walkers = n_walkers
        self._disk_model = anisotropy.DiskModel(model=anisotropy_model)

        self._config = config.load_system_config(system=system)
        self._config_plt = config.load_config(name='plotting')

        plt.rcParams.update(self._config_plt['plt']['rcparams'])

        self.params = self._config['keys']['params']
        self.n_dim = len(self.params)
        self._kepler_radius = self._config['grid']['kepler_radius']

        self.param_labels = []
        self._idx = {}
        for i, param in enumerate(self.params):
            self.param_labels += [self._config_plt['strings']['params'][param]]
            self._idx[param] = i

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

        self.n_autocorr = int(self.n_steps / self.tau.mean())

        print('Unpacking chain')
        self.chain = self._unpack_chain()
        self.derived = self._get_derived()

        self.n_samples = len(self.chain)
        self._cc = chainconsumer.ChainConsumer()
        self._cc.add_chain(self.chain, parameters=self.param_labels)

        self._cc.configure(kde=False,
                           smooth=0,
                           sigmas=np.linspace(0, 2, 5),
                           summary=False,
                           usetex=False)

        self.summary = self._get_summary_stats()

        self.stats = {'total steps': self.n_steps,
                      'autocorr steps': self.n_autocorr,
                      'mean tau': int(self.tau.mean()),
                      'discard': self.discard,
                      'thin': self.thin,
                      'used steps': int(self.n_samples / self.n_walkers),
                      'used samples': self.n_samples,
                      }

        self.print_summary()

    # ===============================================================
    #                      Analysis
    # ===============================================================
    def print_summary(self):
        """Print summary statistics for 1D marginilized posteriors
        """
        chain_title = 'Chain stats'
        print(f'\n{chain_title}\n' + len(chain_title)*'-')
        max_len = len(max(self.stats.keys(), key=len))

        for key, val in self.stats.items():
            print(f'{key.ljust(max_len)} = {val}')

        title = 'Max likelihood estimates'
        print(f'\n{title}\n' + len(title)*'-')

        max_len = len(max(self.params, key=len))

        for param, summ in self.summary.items():
            if None in summ:
                print(f'{param.ljust(max_len)} = unconstrained!')
            else:
                val = summ[1]
                minus = val - summ[0]
                plus = summ[2] - val

                print(f'{param.ljust(max_len)} = {val:.3f}  +{plus:.3f}  -{minus:.3f}')

    def _unpack_chain(self):
        """Unpacks chain of samples from full MCMC chain

        Returns: [n_samples, n_dim]
        """
        chain = self._backend.get_chain(flat=True,
                                        discard=self.discard,
                                        thin=self.thin)
        return chain

    def _get_derived(self):
        """Calculates derived quantities from sample chain

        Returns: {derived_param: [n_samples]}
        """
        print('Calculating derived quantities')
        mass_nw = gravity.mass_from_g(g_nw=self.chain[:, self._idx['g']],
                                      r_nw=self._kepler_radius)

        phi = self.chain[:, self._idx['mass']] / mass_nw

        r_ratio = gravity.get_xi(r_nw=self._kepler_radius,
                                 m_nw=mass_nw,
                                 phi=phi)

        redshift = gravity.redshift_from_xi_phi(phi=phi, xi=r_ratio)

        xi_ratio = self.chain[:, self._idx['xi_ratio']]
        inclination = self._disk_model.inclination(xi=xi_ratio, var='xi_p/xi_b')
        xi_b = self._disk_model.anisotropy(inc=inclination, var='xi_b')
        xi_p = self._disk_model.anisotropy(inc=inclination, var='xi_p')

        derived = {'mass_nw': mass_nw,
                   'radius': phi * self._kepler_radius,
                   'r_ratio': r_ratio,
                   'redshift': redshift,
                   'Mdot1': r_ratio * self.chain[:, self._idx['mdot1']],
                   'Mdot2': r_ratio * self.chain[:, self._idx['mdot2']],
                   'Mdot3': r_ratio * self.chain[:, self._idx['mdot3']],
                   'inclination': inclination,
                   'xi_b': xi_b,
                   'xi_p': xi_p,
                   'distance': self.chain[:, self._idx['d_b']] / np.sqrt(xi_b),
                   }

        return derived

    def _get_summary_stats(self):
        """Get marginilized summary statistics from chain

        Returns: {param: [low, peak, high]}
        """
        summ0 = self._cc.analysis.get_summary()
        summary = {}

        for i, val in enumerate(summ0.values()):
            param = self.params[i]
            summary[param] = val

        return summary

    # ===============================================================
    #                      Plotting
    # ===============================================================
    def plot_1d(self,
                filename=None):
        """Plot 1D marginilized posterior distributions

        Parameters
        ----------
        filename : str
        """
        plot = self._cc.plotter.plot_distributions(filename=filename, col_wrap=3)
        return plot

    def plot_2d(self,
                filename=None):
        """Plot 2D marginilized posterior distributions (i.e. corner plot)

        Parameters
        ----------
        filename : str
        """
        plot = self._cc.plotter.plot(filename=filename)
        return plot
