import os
import numpy as np
import pandas as pd


class ObsData:
    """
    Class to hold observation data

    Attributes
    ----------
    system : str
    epochs : [int]
    data : {}
    """

    def __init__(self,
                 system='gs1826',
                 epochs=(1998, 2000, 2007),
                 ):
        """
        Parameters
        ----------
        system : str
        epochs : [int]
        """
        self.system = system
        self.epochs = epochs

        self._table = None
        self.data = None

        self._load_table()
        self._unpack_data()

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
