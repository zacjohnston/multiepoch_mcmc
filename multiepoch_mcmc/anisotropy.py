import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class DiskModel:
    """
    Class for disk anisotropy model

    Attributes
    ----------
    model : str
        name of anisotropy disk model
    table : pd.DataFrame
        table of anisotropy values

    Methods
    -------
    anisotropy(inc, var)
        returns given anisotropy factor at given inclination
    """

    def __init__(self,
                 model='he16_a',
                 ):
        """
        Parameters
        ----------
        model : str
            name of anisotropy disk model
        """
        self.model = model
        self.table = load_table(model)
        self._interps = self._setup_interpolators()

    def anisotropy(self,
                   inc,
                   var,
                   ):
        """Returns anisotropy factor at given inclination

        Parameters
        ----------
        inc : float or ndarray
            inclination [deg]
        var : str
            anisotropy factor, e.g. 'xi_b'
        """
        return self._interps[var](inc)

    def _setup_interpolators(self):
        """Interpolates anisotropy variable from table
        """
        interps = {}

        for var in self.table.columns:
            if var != 'inc':
                interps[var] = interp1d(x=self.table['inc'], y=self.table[var])

        return interps


def load_table(model):
    """Loads anisotropy table from file

    Parameters
    ----------
    model : str
        name of disk anisotropy model
    """
    path = os.path.dirname(__file__)
    filename = f'{model}.csv'
    filepath = os.path.join(path, '..', 'data', 'anisotropy', filename)

    try:
        print(f'Loading anisotropy table: {os.path.abspath(filepath)}')
        table = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f'No table exists for anisotropy model `{model}`')

    # unpack inverse values
    for col in table.columns:
        keys = col.split('/')

        if len(keys) is 2:
            table[keys[1]] = 1 / table[col]

    return table
