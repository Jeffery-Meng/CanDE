from fvecs import *
import sys

data =  fvecs_read(sys.argv[1]).T
to_ivecs(sys.argv[2], np.argsort(data, axis=1))