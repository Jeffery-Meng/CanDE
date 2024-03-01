import numpy as np
from fvecs import *
import sys

path = sys.argv[1]
size = int(sys.argv[2])
dim = int(sys.argv[3])

rng = np.random.default_rng()
data = rng.normal(size=(size, dim))
norms = np.linalg.norm(data, axis=1)
for i in range(size):
    data[i] /= norms[i]
to_fvecs(path, data)