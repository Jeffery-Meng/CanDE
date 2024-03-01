import timeit


from sys import argv
import os, time
import numpy as np
from fvecs import *

dataset = argv[1]
dim_list = {"audio": 192, "mnist":784, "enron":1369, "trevi":4096, 
    "gist":960, "glove":100, "deep":96, "sift":128}
qn  = 200

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



datapath = "/home/gtnetuser/alt/dataset/siftC-train.fvecs"
querypath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftC-test.fvecs"
kernelpath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftS-kernels.fvecs"
gtpath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftC-gt.ivecs"

data = fvecs_read(datapath).T
n = data.shape[1]
print(n)
queries = fvecs_read(querypath, count=qn*(dim_list[dataset]+1)).T
print(queries.shape)
gt = np.zeros((qn, n), dtype=np.int32)

def run_one(qid):
    e2distances_ = euclidean_distances(queries[:, qid], data.T)
    gt[qid] = np.argsort(e2distances_)
    print(qid)
        

for qid in range(qn):
    run_one(qid)

to_ivecs(gtpath, gt)