import numpy as np
import os
import matplotlib.pyplot as plt
import chainconsumer


class MCPlotter:
    """
    Class for plotting MCMC chains
    """

    def __init__(self,
                 system='gs1826'
                 ):
        """
        Parameters
        ----------
        system : str
        """
        self.system = system
