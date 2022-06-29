import os
import sys
from emcee.backends import HDFBackend


def main(filename):
    """Prints steps completed for given MCMC simulation
    """
    path = os.path.dirname(__file__)
    out_path = os.path.join(path, '..', 'output')
    filepath = os.path.join(out_path, filename)

    print(f'Reading: {os.path.abspath(filepath)}')

    sampler = HDFBackend(filepath, read_only=True)
    n_steps = sampler.shape[0]

    print(f'Steps completed: {n_steps}')


if __name__ == '__main__':
    req_args = 2
    n_args = len(sys.argv)

    if n_args != req_args:
        print(f"Must provide filename, e.g. 'python check_progress.py sampler_gs1826.dat'")
        sys.exit(0)

    main(sys.argv[1])
