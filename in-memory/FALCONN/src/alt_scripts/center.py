from fvecs import *

import numpy as np

datasets = ["audio", "mnist", "enron", "trevi", "glove", "gist", "deep", "sift"]

universe = {}
dimensions = {}

"""with open("l2norms.txt", "r") as fin:
    for line in fin.readlines():
        dataset, _, value = tuple(line.split())
        universe[dataset] = float(value)
with open("datasets.txt", "r") as gtin:
    for i, line in enumerate(gtin.readlines()):
        dataset, dim = tuple(line.split())
        dimensions[dataset] = int(dim)"""

def run(dataset):
    raw_file = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/deep-train.fvecs"
    data = fvecs_read(raw_file)
    data = np.array(data, dtype=np.float32)
    #kernel_file = "half_kernels/" + dataset + ".fvecs"
    #kernels = fvecs_read(kernel_file)
    #kernels = kernels.reshape((1000, dimensions[dataset], dimensions[dataset]))

    query_file = "dataset/" + dataset + "-test.fvecs"
    #query = fvecs_read(query_file)

    means = np.mean(data, axis=0)
    maxs = np.max(data, axis=1)
    mins = np.min(data, axis=1)
    centered_data = data - means
    #centered_queries = query - means

    print(means)
    print(maxs)
    print(mins)

    #to_fvecs("dataset/" + dataset + "-trainC.fvecs", centered_data)
    #to_fvecs("dataset/" + dataset + "-testC.fvecs", centered_queries)
    #to_fvecs("dataset/" + dataset + "-means.fvecs", means[np.newaxis, :])
    


run("deep")
