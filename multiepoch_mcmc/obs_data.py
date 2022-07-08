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
    obs_data : {}
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

        self._obs_table = None
        self.obs_data = None

        self._load_obs_table()
        self._unpack_obs_data()

    def _unpack_obs_data(self):
        """Unpacks observed burst data from loaded table
        """
        self.obs_data = self._obs_table.to_dict(orient='list')

        for key, item in self.obs_data.items():
            self.obs_data[key] = np.array(item)

        # ===== Apply bolometric corrections to fper ======
        u_fper_frac = np.sqrt((self.obs_data['u_cbol'] / self.obs_data['cbol']) ** 2
                              + (self.obs_data['u_fper'] / self.obs_data['fper']) ** 2)

        self.obs_data['fper'] *= self.obs_data['cbol']
        self.obs_data['u_fper'] = self.obs_data['fper'] * u_fper_frac

    def _load_obs_table(self):
        """Loads observed burst data from file
        """
        path = os.path.dirname(__file__)
        filename = f'{self.system}.dat'
        filepath = os.path.join(path, '..', 'data', 'obs', self.system, filename)

        print(f'Loading obs table: {os.path.abspath(filepath)}')
        self._obs_table = pd.read_csv(filepath, delim_whitespace=True)
        self._obs_table.set_index('epoch', inplace=True, verify_integrity=True)

        try:
            self._obs_table = self._obs_table.loc[list(self.epochs)]
        except KeyError:
            raise KeyError(f'epoch(s) not found in obs_data table')
