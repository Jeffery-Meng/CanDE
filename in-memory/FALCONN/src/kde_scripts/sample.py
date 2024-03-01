import sys
import os
import numpy as np
import struct
from fvecs import *

rng = np.random.default_rng()

train_path = os.path.join(sys.argv[1], sys.argv[2] + ".fvecs")
num_samples = int(sys.argv[3])
train = fvecs_read(train_path)

train_a = train.shape[0]
idxes = rng.choice(range(train_a), num_samples, replace=False)
train_n = train[idxes, :]
print(train_n.shape)
output_file = os.path.join(sys.argv[1], sys.argv[4] + ".fvecs")
to_fvecs(output_file, train_n)



