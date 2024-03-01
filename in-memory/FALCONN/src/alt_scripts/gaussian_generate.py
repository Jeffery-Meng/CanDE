
import numpy as np
from fvecs import *


rng = np.random.default_rng(142857)

gauss = np.random.normal(0.,1.,(100*130, 130))
to_fvecs("gauss/jl_weights.fvecs", gauss)




