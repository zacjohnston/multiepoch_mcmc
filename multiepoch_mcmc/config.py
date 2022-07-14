import os
import numpy as np
import ast
import configparser

from multiepoch_mcmc import priors


def load_system_config(system):
    """Loads sysetm config settings from file

    Parameters
    ----------
    system : str
    """
    config_dict = load_config(name=system)
    config_dict['lhood']['priors'] = get_priors(config_dict)
    config_dict['grid']['bounds'] = np.array(config_dict['grid']['bounds'])

    return config_dict


def load_config(name):
    """Loads config settings from file

    Parameters
    ----------
    name : str
    """
    path = os.path.dirname(__file__)
    filepath = os.path.join(path, '..', 'config', f'{name}.ini')

    print(f'Loading config: {os.path.abspath(filepath)}')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file for '{name}' not found")

    ini = configparser.ConfigParser()
    ini.read(filepath)

    config_dict = {}
    for section in ini.sections():
        config_dict[section] = {}
        for option in ini.options(section):
            config_dict[section][option] = ast.literal_eval(ini.get(section, option))

    return config_dict


def get_priors(config_dict):
    """Unpacks prior likelihood pdfs for each parameter

    Defaults to flat prior if none provided

    Parameters
    ----------
    config_dict : {}
    """
    prior_dict = {}
    params = config_dict['keys']['params']
    prior_keys = config_dict['lhood']['priors']

    for param in params:
        key = prior_keys.get(param)
        prior_dict[param] = priors.key_map(key)

    return prior_dict
