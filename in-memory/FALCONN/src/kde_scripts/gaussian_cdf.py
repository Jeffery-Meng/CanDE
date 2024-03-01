from scipy.stats import norm
import numpy as np
import sys
from fvecs import *

def print_cdf(path):
    x = np.arange(-5, 5, 1e-6)
    y = norm.cdf(x)
    to_fvecs(path, y)