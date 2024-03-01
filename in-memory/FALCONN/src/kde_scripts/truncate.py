from fvecs import *
import sys

to_fvecs(sys.argv[2], fvecs_read(sys.argv[1])[:1000, :])
