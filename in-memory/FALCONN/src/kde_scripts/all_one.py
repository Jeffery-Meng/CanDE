from fvecs import *
import sys
import numpy as np

def all_one(num, path):
    arr = np.ones((num, 1))
    to_fvecs(path, arr)