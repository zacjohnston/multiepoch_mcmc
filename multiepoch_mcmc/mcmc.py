import numpy as np
import emcee
import time


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
    sampler = emcee.EnsembleSampler(nwalkers=pos.shape[0],
                                    ndim=pos.shape[1],
                                    log_prob_fn=burstfit.lhood,
                                    threads=n_threads)

    return sampler


def run_sampler(sampler,
                pos,
                n_steps,
                print_progress=True):
    """Runs MCMC sampler for given number of steps

    Returns: State

    Parameters
    ----------
    sampler : EnsembleSampler
        MCMC sampler, as returned by setup_sampler()
    pos : [n_walkers, n_dim]
        initial walker positions, as returned by seed_walker_positions()
    n_steps : int
        number of steps to run
    print_progress : bool
        print progress of sampler each step
    """
    t0 = time.time()
    result = None

    for _, result in enumerate(sampler.sample(pos,
                                              iterations=n_steps,
                                              progress=print_progress)):
        pass

    t1 = time.time()
    dtime = t1 - t0
    time_per_step = dtime / n_steps

    n_walkers = pos.shape[0]
    n_samples = n_walkers * n_steps
    time_per_sample = dtime / n_samples

    print(f'Compute time: {dtime:.1f} s ({dtime/3600:.2f} hr)')
    print(f'Time per step: {time_per_step:.1f} s')
    print(f'Time per sample: {time_per_sample:.4f} s')

    return result


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
