import numpy as np
import emcee


def setup_sampler(burstfit,
                  pos,
                  n_threads=1):
    """Initializes MCMC sampler object

    Returns: EnsembleSampler

    Parameters
    ----------
    burstfit : BurstFit
    pos : [n_walkers, n_dim]
    n_threads : int
        number of compute threads to use
    """
    n_walkers = len(pos)
    n_dim = len(pos[0])

    sampler = emcee.EnsembleSampler(nwalkers=n_walkers,
                                    ndim=n_dim,
                                    log_prob_fn=burstfit.lhood,
                                    threads=n_threads)

    return sampler


def seed_walker_positions(x0,
                          n_walkers,
                          mag=1e-3):
    """Generates initial MCMC walker positions

    Walkers are randomly distributed in a "ball"
    around a chosen starting point coordinate `x0`

    Returns: [n_walkers, len(x0)]

    Parameters
    ----------
    x0: [flt]
        coordinates of initial guess, matching length and ordering of `params`
    n_walkers: int
        number of mcmc walkers to use
    mag: flt
        fractional size of ball
    """
    n_dim = len(x0)
    pos = []

    for _ in range(n_walkers):
        factor = 1 + mag * np.random.randn(n_dim)
        pos += [x0 * factor]

    return np.array(pos)
