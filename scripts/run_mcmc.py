import numpy as np
import sys
import os
import time
from multiprocessing import Pool
from emcee import EnsembleSampler, backends

# pyburst
from multiepoch_mcmc import mcmc, burstfit, grid_interpolator
from pyburst.mcmc import mcmc_versions

# =============================================================================
# Usage:
# python run_mcmc.py [source] [version] [n_steps] [n_threads] [save_steps]
# =============================================================================

os.environ["OMP_NUM_THREADS"] = "1"


def main(n_steps,
         n_walkers=1000,
         n_threads=6,
         restart_step=None):
    """Runs an MCMC simulation using interpolated burst model grid

    Parameters
    ----------
    """
    path = os.path.dirname(__file__)
    out_path = os.path.join(path, '..', 'output')
    filepath = os.path.join(out_path, 'sampler.h5')
    backend = backends.HDFBackend(filepath)

    n_threads = int(n_threads)
    n_walkers = int(n_walkers)

    print(f'Using {n_threads} threads')

    # if restart_step is None:
    restart = False
    start = 0
    x0 = [0.086, 0.115, 0.132, 0.702, 0.011, 0.41, 0.2, 0.22, 2.45, 2.1, 6.47, 1.47]
    pos = mcmc.seed_walker_positions(x0, n_walkers=n_walkers)
    # else:
    #     restart = True
    #     start = int(restart_step)
    #     chain0 = mcmc_tools.load_chain(source=source, version=version, n_walkers=n_walkers,
    #                                    n_steps=start)
    #     pos = chain0[:, -1, :]

    mv = mcmc_versions.McmcVersion('grid5', 0)
    interpolator = grid_interpolator.GridInterpolator()

    bfit = burstfit.BurstFit(grid_interpolator=interpolator,
                             priors=mv.priors,
                             grid_bounds=mv.grid_bounds,
                             weights=mv.weights)

    t0 = time.time()

    # ===== Run MCMC sampler =====
    with Pool(processes=n_threads) as pool:
        sampler = EnsembleSampler(nwalkers=pos.shape[0],
                                  ndim=pos.shape[1],
                                  log_prob_fn=bfit.lhood,
                                  pool=pool,
                                  backend=backend)

        sampler.run_mcmc(initial_state=pos,
                         nsteps=n_steps,
                         progress=True)

        # ===== concatenate loaded chain to current chain =====
        # if restart:
        #     save_chain = np.concatenate([chain0, sampler.chain], 1)
        # else:
        #     save_chain = sampler.chain

    print('=' * 30)
    print('Done!')

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

    if n_args < min_args:
        print(f"""Must provide at least {min_args} parameter(s):
                    1. n_steps       : number of mcmc steps to take
                    (2. save_steps   : steps to do between saves)
                    (3. n_walkers    : number of mcmc walkers)
                    (4. n_threads    : number of threads/cores to use)
                    (5. restart_step : step to restart from)""")
        sys.exit(0)

    if n_args == min_args:
        main(int(sys.argv[1]))
    else:
        main(int(sys.argv[1]), **dict(arg.split('=') for arg in sys.argv[2:]))



