import os
import pandas as pd


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

    print(f'Loading anisotropy table: {os.path.abspath(filepath)}')
    table = pd.read_csv(filepath)

    return table
