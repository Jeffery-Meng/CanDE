from fvecs import *

vec = np.empty((100000, 1), dtype=np.int32)
for i in range(100000):
    vec[i] = i
to_ivecs("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/identity.ivecs", vec)