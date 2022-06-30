import sys
import os
import time
from multiprocessing import Pool
from emcee import EnsembleSampler, backends

# pyburst
from multiepoch_mcmc import mcmc, burst_sampler

# =============================================================================
# Usage:
# python run_mcmc.py <n_steps>
# =============================================================================

os.environ["OMP_NUM_THREADS"] = "1"  # recommended for parallel emcee


def main(n_steps,
         n_walkers=1000,
         n_threads=4,
         system='gs1826',
         restart=False,
         progress=True):
    """Runs an MCMC simulation using interpolated burst model grid

    Parameters
    ----------
    n_steps : int
    n_walkers : int
    system : str
    n_threads : int
    restart : bool
    progress : bool
    """
    bool_map = {'True': True, 'False': False}

    n_threads = int(n_threads)
    n_walkers = int(n_walkers)
    restart = bool_map[str(restart)]
    progress = bool_map[str(progress)]

    filename = f'sampler_{system}_{n_walkers}w.h5'
    path = os.path.dirname(__file__)
    out_path = os.path.join(path, '..', 'output')
    filepath = os.path.join(out_path, filename)

    backend = backends.HDFBackend(filepath)

    bsampler = burst_sampler.BurstSampler(system=system)
    n_dim = len(bsampler.params)

    t0 = time.time()
    print(f'\nRunning {n_walkers} walkers for {n_steps} steps using {n_threads} threads')
    print(f'Output file: {os.path.abspath(filepath)}')

    if restart:
        print(f'Restarting from step {backend.iteration}\n')
        pos = None
    else:
        backend.reset(nwalkers=n_walkers, ndim=n_dim)
        pos = mcmc.seed_walker_positions(x_start=bsampler.x_start,
                                         n_walkers=n_walkers)

    with Pool(processes=n_threads) as pool:
        sampler = EnsembleSampler(nwalkers=n_walkers,
                                  ndim=n_dim,
                                  log_prob_fn=bsampler.lhood,
                                  pool=pool,
                                  backend=backend)

        sampler.run_mcmc(initial_state=pos,
                         nsteps=n_steps,
                         progress=progress)

    print('\nDone!')

    t1 = time.time()
    dt = t1 - t0
    time_per_step = dt / n_steps
    time_per_sample = dt / (n_walkers * n_steps)

    print(f'Total compute time: {dt:.0f} s ({dt/3600:.2f} hr)')
    print(f'Average time per step: {time_per_step:.1f} s')
    print(f'Average time per sample: {time_per_sample:.4f} s')


if __name__ == "__main__":
    min_args = 1
    n_args = len(sys.argv)

    if n_args < min_args + 1:
        print(f"""Must provide at least {min_args} parameter(s):
                    1. n_steps       : number of mcmc steps to take
                    (2. n_walkers    : number of mcmc walkers)
                    (3. n_threads    : number of threads/cores to use)
                    (4. system       : name of bursting system)""")
        sys.exit(0)

    if n_args == min_args:
        main(int(sys.argv[1]))
    else:
        main(int(sys.argv[1]), **dict(arg.split('=') for arg in sys.argv[min_args+1:]))



