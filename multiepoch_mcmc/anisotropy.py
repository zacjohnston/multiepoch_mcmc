import os
import pandas as pd


def load_table(model):
    """Loads anisotropy table from file

    Parameters
    ----------
    model : str
        name of anisotropy model
    """
    path = os.path.dirname(__file__)
    filename = f'{model}.csv'
    filepath = os.path.join(path, '..', 'data', 'anisotropy', filename)

    print(f'Loading anisotropy table: {os.path.abspath(filepath)}')
    table = pd.read_csv(filepath)

    return table
