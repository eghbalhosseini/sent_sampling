import sys

import numpy as np
import logging

LOG_BASE = np.e
EPSILON = 1e-16
PRECISION = 1e-200

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)

#‌impor functions from Noga's optimization pipeline.
sys.path.insert(1,'/Users/eghbalhosseini/MyCodes/opt-exp-design-nlp/')
from opt_exp_design import coordinate_ascent, D


class optim:
    def __init__(self, n_init=1, n_iter=200, objective_function=D, optim_algorthim=coordinate_ascent):
        self.n_iter=n_iter
        self.n_init=n_init
        self.objective_function=objective_function
        self.optim_algorithm=optim_algorthim
    def __call__(self, *args, **kwargs):
        pass




