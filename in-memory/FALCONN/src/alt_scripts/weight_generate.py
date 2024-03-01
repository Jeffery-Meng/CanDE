
import numpy as np
from fvecs import *


np.random.seed(142857)


# %%
weight_data = np.random.uniform(0,1,(100,128))
to_fvecs("dataset/l2weights.fvecs", weight_data)

# %%
weight_data.shape


# %%
def euclidean_distances(X, Y):
    if X.ndim < 2:
        X = X.reshape(1,X.size)
    if Y.ndim < 2:
        Y = Y.reshape(1,Y.size)



    e_dist = X[:, np.newaxis] - Y
    e_dist **= 2
    e_dist = np.sum(e_dist, axis=2)
    e_dist = np.power(e_dist, 0.5)

    return e_dist




