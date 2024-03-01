from fvecs import *
import sys

to_fvecs(sys.argv[2], np.abs(fvecs_read(sys.argv[1])))
