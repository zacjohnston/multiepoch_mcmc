import numpy as np


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
        initial guess coordinates, matching length and ordering of `params`
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
