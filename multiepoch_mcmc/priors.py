import numpy as np
from scipy.stats import norm, beta


def key_map(key):
    """Maps key to prior function

    Parameters
    ----------
    key : str
    """
    keys = {None: flat_prior,
            'z_beta': z_beta_prior,
            }

    prior = keys.get(key)

    if prior is None:
        raise ValueError(f"invalid prior key '{key}' in config")
    return


def flat_prior(x):
    """Returns flat (uniform) prior likelihood
    """
    return 1


def gaussian(mean, std):
    """Returns callable function for Gaussian distribution
    """
    return norm(loc=mean, scale=std).pdf


def z_beta_prior(z,
                 z_sun=0.01,
                 a=10.1,
                 b=3.5,
                 loc=-3.5,
                 scale=4.5,
                 ):
    """Returns prior likelihood on metallicity

    Uses beta distribution based on modelled galactic composition
    in the direction of GS-1826

    For details see Section 2.9 of Johnston et al. (2020)
    """
    log_beta = beta(a=a, b=b, loc=loc, scale=scale).pdf

    logz = np.log10(z / z_sun)
    lnprior = log_beta(logz)

    return lnprior
