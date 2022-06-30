import sys
import os

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
    # ===== parse args =====
    bool_map = {'True': True, 'False': False}
    n_threads = int(n_threads)
    n_walkers = int(n_walkers)
    restart = bool_map[str(restart)]
    progress = bool_map[str(progress)]

    # ===== Setup burst sampler =====
    bsampler = burst_sampler.BurstSampler(system=system)

    # ===== Run MCMC simulation =====
    mcmc.run_sampler_pool(bsampler=bsampler,
                          n_walkers=n_walkers,
                          n_threads=n_threads,
                          n_steps=n_steps,
                          restart=restart,
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



