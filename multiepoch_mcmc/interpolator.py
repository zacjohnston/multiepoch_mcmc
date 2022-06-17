import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import time


class Interpolator:
    """Interpolates burst properties from model grid

    parameters
    ----------
    grid_file : str
        filename of model table in `data/model_grid/`
    params : [str]
    bvars : [str]
    """

    def __init__(self,
                 grid_file='johnston_2020.txt',
                 params=('mdot', 'x', 'z', 'qb', 'g'),
                 bvars=('rate', 'u_rate', 'energy', 'u_energy', 'peak', 'u_peak'),
                 ):
        path = os.path.dirname(__file__)
        filepath = os.path.join(path, '..', 'data', 'model_grid', grid_file)
        self.filepath = os.path.abspath(filepath)

        self.params = params
        self.bvars = bvars
        self.grid = None
        self._interpolator = None

        self.load_grid()
        self.setup_interpolator()

    def load_grid(self):
        """Loads model grid table
        """
        print(f'Loading grid: {self.filepath}\n')
        self.grid = pd.read_csv(self.filepath, delim_whitespace=True)

        print('Grid parameters\n' + 15*'-')
        for p in self.params:
            print(f'{p.ljust(5)} = {np.unique(self.grid[p])}')

        print(f'\nTotal models: {len(self.grid)}')

    def setup_interpolator(self):
        """Creates interpolator function from grid
        """
        print('\nGenerating interpolator\n' + 23*'-')
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

    def interpolate(self, x):
        """Interpolate burst variables from grid

        Returns: [flt]
            Interpolated burst variables at point 'x'
            Matches length and ordering of 'bvars'

        parameters
        ----------
        x : [flt]
            Coordinates of point to interpolate
            Must exactly match length and ordering of 'params'
        """
        return self._interpolator(x)
