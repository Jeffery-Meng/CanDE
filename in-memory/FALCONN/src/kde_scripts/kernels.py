from math import *
import numpy as np
from sklearn.metrics import pairwise_distances

def gen_kernel(X_test,X_train,weight_vec):
    distances = pairwise_distances(X_train,X_test,n_jobs=-1)
    gamma = 1/27
    kernel_result= np.full(len(X_test),0)

    for i in range(len(X_test)):
        if i % 10 == 0:
            print(i)
        sum_temp = 0
        for j in range(len(X_train)):
            sum_temp += weight_vec[j]*exp(-gamma*distances[j,i])
        kernel_result[i] = sum_temp

    return kernel_result