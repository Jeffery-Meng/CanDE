import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from sympy import beta
from fvecs import *
from math import *

datasets = ["audio", "mnist", "enron", "trevi", "glove", "gist", "deep", "sift"]

universe = {}
dimensions = {}

with open("l2norms.txt", "r") as fin:
    for line in fin.readlines():
        dataset, _, value = tuple(line.split())
        universe[dataset] = float(value)
with open("datasets.txt", "r") as gtin:
    for i, line in enumerate(gtin.readlines()):
        dataset, dim = tuple(line.split())
        dimensions[dataset] = int(dim)


def alt_dist(kernel, query, x):
    return np.linalg.norm(kernel@ x - query)

def l2_dist( query, x):
    return np.linalg.norm(x - query)

def run(dataset, id):
    univ = universe[dataset] + 1 # consider the added 1 dimension.
    gt_file = "ground_truth/" + dataset + ".txt"
    gt_file2 = "ground_truth/" + dataset + "2.txt"
    nns = np.empty((1000, 100), dtype=np.int32)
    alphas = np.empty(1000)
    betas = np.empty(1000)
    dists = np.empty(1000)
    ratios = np.empty(1000)
    
    with open(gt_file, "r") as gtin:
        for i, line in enumerate(gtin.readlines()[:100]):
            nns[i] = [int(x) for x in line.split()[:100]]

    with open(gt_file2, "r") as gtin:
        for i, line in enumerate(gtin.readlines()):
            nns[i+100] = [int(x) for x in line.split()[:100]]
    
    kernel_file = "half_kernels/" + dataset + ".fvecs"
    kernels = fvecs_read(kernel_file)
    kernels = kernels.reshape((1000, dimensions[dataset], dimensions[dataset]))

    query_file1 = "dataset/" + dataset + "-ALTtest.fvecs"
    query_file2 = "dataset/" + dataset + "-ALTtest2.fvecs"

    raw_file = "dataset/" + dataset + "-train.fvecs"
    data = fvecs_read(raw_file)

    read_count = 100 * (dimensions[dataset]+1)  * 4
    queries = fvecs_read(query_file1, count=read_count)[:100]
    query2 = fvecs_read(query_file2)
    queries = np.concatenate((queries, query2))

    print(alt_dist(kernels[id], queries[id].T, data[nns[id,0]].T))
    print(alt_dist(kernels[id], queries[id].T, data[nns[id,1]].T))
    print(alt_dist(kernels[id], queries[id].T, data[nns[id,2]].T))
    print(alt_dist(kernels[id], queries[id].T, data[0].T))
    print(alt_dist(kernels[id], queries[id].T, data[1000].T))
    print(alt_dist(kernels[id], queries[id].T, data[10000].T))
    print(np.linalg.norm(queries[id]), np.linalg.norm(kernels[id].T@kernels[id]),
     np.linalg.norm(kernels[id]@data[10000].T),np.linalg.norm(kernels[id]@data[10000].T - queries[id].T) )

    print(data.mean(axis = 0))
    print(data.min(axis = 0), data.max(axis=0))
run("audio", 897)