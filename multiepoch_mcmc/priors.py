import numpy as np
from scipy.stats import beta


def flat_prior(x):
    return 1


def log_z_prior(z,
                z_sun=0.01,
                a=10.1,
                b=3.5,
                loc=-3.5,
                scale=4.5,
                ):
    """Returns prior on log metallicity
    """
    log_beta = beta(a=a, b=b, loc=loc, scale=scale).pdf

    logz = np.log10(z / z_sun)
    lnprior = log_beta(logz)

    return lnprior
