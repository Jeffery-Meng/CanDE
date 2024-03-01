import numpy as np
import sys
from fvecs import *

data = fvecs_read(sys.argv[1])
query = fvecs_read(sys.argv[2])
data = data.astype(np.float64)
query = query.astype(np.float64)
data_output = sys.argv[3]
query_output = sys.argv[4]
e = 1e-6

stdev = np.std(data, axis=0)
data = data[:, stdev > e]
query = query[:, stdev > e]
stdev = stdev[stdev > 0]

n, dim = data.shape
for i in range(n):
    data[i, :] /= stdev
qn, dim = query.shape
for i in range(qn):
    query[i, :] /= stdev

to_fvecs(data_output, data)
to_fvecs(query_output, query)
