import os
import pandas as pd
from scipy.interpolate import interp1d


class DiskModel:
    """
    Class for disk anisotropy model
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


def load_table(model,
               components=('p', 'b', 'd', 'r'),
               ):
    """Loads anisotropy table from file

    Parameters
    ----------
    model : str
        name of disk anisotropy model
    components : [str]
        list of anisotropy components (by suffix letter)
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
    for c in components:
        table[f'xi_{c}'] = 1 / table[f'1/xi_{c}']

    return table
