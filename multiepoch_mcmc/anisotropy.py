import os
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
        returns given anisotropy factor for given inclination
    inclination(xi, var)
        returns inclination corresponding to given xi factor
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

    def anisotropy(self,
                   inc,
                   var,
                   ):
        """Returns anisotropy factor for given inclination

        Returns : float or ndarray
            returns np.nan if no solution

        Parameters
        ----------
        inc : float or ndarray
            inclination [deg]
        var : str
            name of anisotropy factor, e.g. 'xi_b'
        """
        interp = interp1d(x=self.table['inc'],
                          y=self.table[var],
                          bounds_error=False)
        xi = interp(inc)

        return xi

    def inclination(self,
                    xi,
                    var
                    ):
        """Returns inclination corresponding to given xi factor

        Returns : float or ndarray
            returns np.nan if no solution

        Parameters
        ----------
        xi : float or ndarray
            xi factor value
        var : str
            name of anisotropy factor, e.g. 'xi_b'
        """
        interp = interp1d(x=self.table[var],
                          y=self.table['inc'],
                          bounds_error=False)

        inc = interp(xi)

        return inc


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

    table['xi_b/xi_p'] = table['xi_b'] / table['xi_p']

    return table
