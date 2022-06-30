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
         n_walkers=1024,
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

    bsampler = burst_sampler.BurstSampler(system=system)
    n_dim = len(bsampler.params)

    print(f'\nRunning {n_walkers} walkers for {n_steps} steps using {n_threads} threads')

    backend = mcmc.open_backend(system=system, n_walkers=n_walkers)

    if restart:
        print(f'Restarting from step {backend.iteration}\n')
        pos = None
    else:
        backend.reset(nwalkers=n_walkers, ndim=n_dim)
        pos = mcmc.seed_walker_positions(x_start=bsampler.x_start,
                                         n_walkers=n_walkers)

    t0 = time.time()

    mcmc.run_sampler_pool(bsampler=bsampler,
                          n_walkers=n_walkers,
                          n_threads=n_threads,
                          n_steps=n_steps,
                          pos=pos,
                          progress=progress)



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



