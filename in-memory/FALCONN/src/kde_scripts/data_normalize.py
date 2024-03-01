import sys
from fvecs import *
import numpy as np

data = fvecs_read(sys.argv[1])
data = data.astype(np.float64)
queries = fvecs_read(sys.argv[3])
stdev = np.std(data, axis=0)

def compute_factor(val):
    if val == 0:
        return 1.0
    else:
        return 1.0 / val

factors = np.vectorize(compute_factor)(stdev)

# for i in range(len(factors)):
#     print(np.std(data[:, i]))
#     data[:, i] *= factors[i]
#     print(factors[i], np.std(data[:, i]))
#  exit(1)
for row in data:
    row *= factors
for row in queries:
    row *= factors
print(np.std(data, axis=0))
print(np.std(queries, axis=0))
to_fvecs(sys.argv[2], data)
to_fvecs(sys.argv[4], queries)



