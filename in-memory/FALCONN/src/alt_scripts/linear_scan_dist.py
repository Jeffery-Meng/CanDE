import timeit


from sys import argv
import os, time
import numpy as np
from fvecs import *
from plaintext import *



def read_transform(dim, qid, transform):
    count = dim * (dim + 1)
    offset = count * 4 * qid
    return fvecs_read(transform, count=count, offset=offset)

"""Returns squared euclidean distances."""
def euclidean_distances(X, Y):
    if X.ndim < 2:
        X = X.reshape(1,X.size)
    if Y.ndim < 2:
        Y = Y.reshape(1,Y.size)



    e_dist = X[:, np.newaxis] - Y
    e_dist **= 2
    e_dist = np.sum(e_dist, axis=2)
    #e_dist = np.power(e_dist, 0.5)

    return e_dist

def run_one(qid, data, query, kernel, gt, dist):
    transformed_data = kernel @ data
    transformed_query = kernel @ query

    e2distances_ = euclidean_distances(transformed_query, transformed_data.T)
    gt[qid] = np.argsort(e2distances_)
    dist[qid] = np.sort(e2distances_)
    print(qid)

""" Linear Scan function to be called externally
The config is passed as a dictionary "info"
"""
def linear_scan_ext(info, data, queries, kernels):
    n = data.shape[0] 
    qn = queries.shape[0]
    
    gt = np.zeros((qn, n), dtype=np.int32)
    dist = np.zeros((qn, n), dtype=np.float32)
    data = data.T
    queries = queries.T
 
    for qid in range(qn):
        run_one(qid, data, queries[:, qid], kernels[qid], gt, dist)
    
    MaxK = 100
    to_ivecs(info["gt"], gt)
    print_gt(info["gt-plain"], gt[:, :MaxK], dist[:, :MaxK])
