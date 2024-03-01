from fvecs import *
import sys
import numpy as np

data =  fvecs_read(sys.argv[1])
query = fvecs_read(sys.argv[2])
print(np.linalg.norm(data[int(sys.argv[3]), :] - query[int(sys.argv[4]), :]))