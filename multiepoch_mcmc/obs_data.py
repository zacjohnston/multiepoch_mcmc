import os
import numpy as np
import pandas as pd


class ObsData:
    """
    Class to hold observation data

    Attributes
    ----------
    bvars : [str]
        list of burst variables to use
    epochs : [int]
        list of epoch years to use
    data : {}
        observation data by bvar key
    system : str
        name of bursting system
    y : [n_epochs, n_bvars]
        2D array of burst properties vs. epoch
    u_y : [n_epochs, n_bvars]
        corresponding uncertainties to `y`
    """

    def __init__(self,
                 system='gs1826',
                 epochs=(1998, 2000, 2007),
                 bvars=('rate', 'fluence', 'peak', 'fper', 'fedd'),
                 ):
        """
        Parameters
        ----------
        system : str
        epochs : [int]
        bvars : [str]
        """
        self.system = system
        self.epochs = epochs
        self.bvars = bvars

        self._table = None
        self.data = None

        self.y = np.full([len(epochs), len(bvars)], np.nan)
        self.u_y = np.full_like(self.y, np.nan)

        self._load_table()
        self._unpack_data()
        self._fill_epoch_array()

    def _fill_epoch_array(self):
        """Fills epoch array with burst variables
        """
        for i, bvar in enumerate(self.bvars):
            self.y[:, i] = self.data[bvar]
            self.u_y[:, i] = self.data[f'u_{bvar}']

    def _unpack_data(self):
        """Unpacks observed burst data from loaded table
        """
        self.data = self._table.to_dict(orient='list')

        for key, item in self.data.items():
            self.data[key] = np.array(item)

        # ===== Apply bolometric corrections to fper ======
        u_fper_frac = np.sqrt((self.data['u_cbol'] / self.data['cbol']) ** 2
                              + (self.data['u_fper'] / self.data['fper']) ** 2)

        self.data['fper'] *= self.data['cbol']
        self.data['u_fper'] = self.data['fper'] * u_fper_frac

    def _load_table(self):
        """Loads observed burst data from file
        """
        path = os.path.dirname(__file__)
        filename = f'{self.system}.dat'
        filepath = os.path.join(path, '..', 'data', 'obs', self.system, filename)

        print(f'Loading obs table: {os.path.abspath(filepath)}')
        self._table = pd.read_csv(filepath, delim_whitespace=True)
        self._table.set_index('epoch', inplace=True, verify_integrity=True)

        try:
            self._table = self._table.loc[list(self.epochs)]
        except KeyError:
            raise KeyError(f'epoch(s) not found in data table')
