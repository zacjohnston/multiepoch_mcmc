import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import time
import pickle


class GridInterpolator:
    """
    A class to quickly interpolate burst variables from a model grid

    Attributes
    ----------
    file : str
        filename of model grid located in 'data/model_grid/'
    grid : pd.DataFrame
        table of model grid data
    params : [str]
        list of model parameters
    bvars : [str]
        list of burst variables

    Methods
    -------
    interpolate(x)
        Interpolates burst variables at given grid coordinate
    """

    def __init__(self,
                 file='johnston_2020.txt',
                 params=('mdot', 'qb', 'x', 'z', 'g'),
                 bvars=('rate', 'u_rate', 'energy', 'u_energy', 'peak', 'u_peak'),
                 reconstruct=True,
                 ):
        """
        Parameters
        ----------
        file : str
            filename of model table in `data/model_grid/`
        params : [str]
            list of input model parameters
        bvars : [str]
            list of output burst variables
        reconstruct : bool
            reconstruct interpolator object (as opposed to loading from file)
        """
        path = os.path.dirname(__file__)
        gridpath = os.path.join(path, '..', 'data', 'model_grid', file)

        int_file = 'interpolator.pickle'
        intpath = os.path.join(path, '..', 'data', 'temp', int_file)

        self.file = file
        self._gridpath = os.path.abspath(gridpath)
        self._intpath = os.path.abspath(intpath)

        self.params = params
        self.bvars = bvars
        self.grid = None
        self._interpolator = None

        self._load_grid()

        if reconstruct:
            self._setup_interpolator()
        else:
            self._load_interpolator()

    def interpolate(self, x):
        """Interpolates burst variables from grid

        Returns: [flt]
            Interpolated burst variables at point 'x'
            Matches length and ordering of 'bvars'

        Parameters
        ----------
        x : [flt]
            Coordinates of grid point to interpolate
            Must exactly match length and ordering of 'params'
        """
        return self._interpolator(x)

    # ===============================================================
    #                      Setup
    # ===============================================================
    def _load_grid(self):
        """Loads model grid from file
        """
        print(f'Loading grid: {self._gridpath}')
        self.grid = pd.read_csv(self._gridpath, delim_whitespace=True)

        print('\nGrid parameters\n' + 15*'-')
        for p in self.params:
            print(f'{p.ljust(5)} = {np.unique(self.grid[p])}')

        print(f'\nTotal models: {len(self.grid)}')

    def _setup_interpolator(self):
        """Generates interpolator object from grid
        """
        print('\nGenerating grid interpolator')
        t0 = time.time()
        x = []

        for p in self.params:
            x += [np.array(self.grid[p])]

        n_models = len(self.grid)
        n_bvars = len(self.bvars)

        x = tuple(x)
        y = np.full((n_models, n_bvars), np.nan)

        for i, var in enumerate(self.bvars):
            y[:, i] = np.array(self.grid[var])

        self._interpolator = LinearNDInterpolator(x, y)

        t1 = time.time()
        print(f'Setup time: {t1-t0:.1f} s')

    def _save_interpolator(self):
        """Saves interpolator to file
        """
        pickle.dump(self._interpolator, open(self._intpath, 'wb'))

    def _load_interpolator(self):
        """Loads interpolator from file
        """
        print(f'Loading interpolator: {self._intpath}')
        self._interpolator = pickle.load(open(self._intpath, 'rb'))
