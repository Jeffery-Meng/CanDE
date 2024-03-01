import timeit


from sys import argv
import os, time
import numpy as np
from fvecs import *

dataset = argv[1]
dim_a = int(argv[2])
dim_list = {"audio": 192, "mnist":784, "enron":1369, "trevi":4096, 
    "gist":960, "glove":100, "deep":96, "sift":128}
qn  = 100

def read_transform(dataset, qid, transform):
    count = dim_list[dataset] * (dim_list[dataset] + 1)
    offset = count * 4 * qid
    return fvecs_read(transform, count=count, offset=offset)

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

data_dict = {"sift": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftC-train.fvecs",
"deep" : "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/deep-train.fvecs",
"audio" : "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/audioC-train.fvecs"}

datapath = data_dict[dataset]

querypath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/subspace/" + \
    dataset + str(dim_a) + "-queries.fvecs"
kernelpath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/subspace/" + \
    dataset + str(dim_a) + "-kernels.fvecs"
gtpath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/subspace/" + \
    dataset + str(dim_a) + "-gt.ivecs"

data = fvecs_read(datapath).T
n = data.shape[1]
print(n)
queries = fvecs_read(querypath).T
gt = np.zeros((qn, n), dtype=np.int32)
def run_one(qid):
    kernel = read_transform(dataset, qid, kernelpath)
    transformed_data = kernel @ data
    transformed_query = kernel @ queries[:, qid]

    e2distances_ = euclidean_distances(transformed_query, transformed_data.T)
    gt[qid] = np.argsort(e2distances_)
    print(qid)
        

for qid in range(qn):
    run_one(qid)

to_ivecs(gtpath, gt)