import os
import numpy as np
from emcee import EnsembleSampler, backends
import time


def setup_sampler(bsampler,
                  n_walkers,
                  n_dim,
                  backend=None,
                  pool=None):
    """Initializes MCMC sampler object

    Returns: EnsembleSampler

    Parameters
    ----------
    bsampler : BurstSampler
    n_walkers : int
    n_dim : int
    backend : HDFBackend
    pool : multiprocessing.Pool
        used for parallel compute
    """
    sampler = EnsembleSampler(nwalkers=n_walkers,
                              ndim=n_dim,
                              log_prob_fn=bsampler.lhood,
                              backend=backend,
                              pool=pool)
    return sampler


def run_sampler(sampler,
                pos,
                n_steps,
                print_progress=True):
    """Runs MCMC sampler for given number of steps

    Returns: State
        sampler state at final step
        Not actually needed; full data is held in sampler object

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

    result = sampler.run_mcmc(initial_state=pos,
                              nsteps=n_steps,
                              progress=print_progress)

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


def seed_walker_positions(x_start,
                          n_walkers,
                          sigma_frac=1e-3):
    """Generates initial MCMC walker positions

    Walker positions are randomly distributed within a Gaussian n-ball,
    centred on a given initial guess `x_start`

    Returns: ndarray
        shape (n_walkers, n_dim)

    Parameters
    ----------
    x_start: [flt]
        coordinates of initial guess, matching length and ordering of `params`
    n_walkers: int
        number of mcmc walkers to use
    sigma_frac: flt
        fractional standard deviation of Gaussian n-ball
    """
    n_dim = len(x_start)
    pos = []

    for _ in range(n_walkers):
        factor = 1 + sigma_frac * np.random.randn(n_dim)
        pos += [x_start * factor]

    return np.array(pos)


def open_backend(system,
                 n_walkers):
    """Returns sampler output file for reading/writing

    Returns HDFBackend

    Parameters
    ----------
    system : str
    n_walkers : int
    """
    filename = f'sampler_{system}_{n_walkers}w.h5'
    path = os.path.dirname(__file__)
    out_path = os.path.join(path, '..', 'output')

    filepath = os.path.join(out_path, filename)
    print(f'Output file: {os.path.abspath(filepath)}')

    backend = backends.HDFBackend(filepath)

    return backend
