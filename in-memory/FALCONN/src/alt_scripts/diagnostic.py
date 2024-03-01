from fvecs import *
from l2_mapped import *


import numpy as np
from numpy.linalg import norm
from math import sqrt

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
universe["audio"] = 100000

def diagnostic(data, query, kernel):
    #kernel = np.eye(data.shape[0])
    print(norm(data), norm(query), norm(data-query), norm(kernel@(data-query)), norm(kernel))

def diagn(data, queries, kernels, xid, qid):
    diagnostic(data[xid], queries[qid], kernels[qid])

def get_c(kernel, dim):
    return norm(kernel) ** 2 / dim

def distance_func(kernel, data, c):
    kernel_comp = kernel.T @ kernel-c * np.eye(kernel.shape[1])
    print(norm(data.T @ data), norm(kernel_comp),
     np.sum(kernel_comp * (data.T @ data)))
    return norm( data.T @ data + kernel_comp)

def compute_beta_c(kernel, c, universe):
    #print(norm(kernel.T@kernel), norm(kernel.T@kernel - c * np.eye(kernel.shape[1])))
    return universe / sqrt(norm(kernel.T@kernel - c * np.eye(kernel.shape[1])))

def compute_beta_3(kernel, dim, universe):
    temp =  (dim * norm(kernel.T@kernel) ** 2 - norm(kernel)** 4) / (dim-1)
    return temp ** (1/4) / universe

def compute_c_2(kernel, dim, universe, beta):
    return (norm(kernel)**2 + beta**2*universe**2) / dim

def run2(dataset):
    raw_file = "dataset/" + dataset + "-train.fvecs"
    data = fvecs_read(raw_file)
    gt_file = "ground_truth/" + dataset + ".txt"
    gt_file2 = "ground_truth/" + dataset + "2.txt"
    #kernel_file = "half_kernels/" + dataset + ".fvecs"
    #kernels = fvecs_read(kernel_file)
    #kernels = kernels.reshape((1000, dimensions[dataset], dimensions[dataset]))

    query_file = "dataset/" + dataset + "-test.fvecs"
    queries = fvecs_read(query_file)

    kernel_file = "half_kernels/" + dataset + ".fvecs"
    kernels = fvecs_read(kernel_file)
    kernels = kernels.reshape((1000, dimensions[dataset], dimensions[dataset]))

    nns = np.empty(1000, dtype=np.int32)
    c_coef = np.empty(1000)

    with open(gt_file, "r") as gtin:
        for i, line in enumerate(gtin.readlines()):
            nns[i] = int(line.split()[0])

    with open(gt_file2, "r") as gtin:
        for i, line in enumerate(gtin.readlines()):
            nns[i+100] = int(line.split()[0])

    betas = np.empty(1000)
    alpha_rot = universe[dataset] * dimensions[dataset] ** (-1/6)
    data_alpha = np.empty((data.shape[0], dimensions[dataset] + 2))
    data_alpha[:, :-2] = data / alpha_rot
    data_alpha[:, -2] = -1
    kernel_alpha = np.zeros((1000, dimensions[dataset], dimensions[dataset] + 2))
    normalized_u = (1 + (universe[dataset]/alpha_rot)**2) ** 0.5
    for i, point in enumerate(data_alpha):
        if norm(point) > normalized_u:
            data_alpha[i] = np.array([100000] * (dimensions[dataset] + 2))
        else:
            data_alpha[i, -1] = sqrt(normalized_u**2 - norm(point)**2) 
    for i in range(1000):
        kernel_alpha[i, :, :-2] = kernels[i]
        kernel_alpha[i, :, -2] = kernels[i] @ queries[i].T / alpha_rot
        #print(norm(queries[i])/ alpha_rot)
        #kernel_alpha[i, :, :-2] = np.eye(dimensions[dataset])
        #kernel_alpha[i, :, -2] = queries[i].T / alpha_rot

        betas[i] = compute_beta_3(kernel_alpha[i], dimensions[dataset] + 2, normalized_u)
        c_coef[i] = compute_c_2(kernel_alpha[i], dimensions[dataset] + 2, normalized_u, betas[i])
        old_c = get_c(kernel_alpha[i], dimensions[dataset] + 2)
        old_beta = compute_beta_c(kernel_alpha[i], old_c, normalized_u)
        old_c *= old_beta **2
        #print("c", c_coef[i])
        betas[i] = 1 / betas[i]
        c_coef[i] *= betas[i]**2
        #print("bc", betas[i], c_coef[i], old_beta, old_c)
        kernel_alpha[i] *= betas[i]
        #print(normalized_u, norm(kernel_alpha[i].T@kernel_alpha[i] - c_coef[i]* np.eye(dimensions[dataset]+2)))

    qid = 419
    print("bc", betas[qid], c_coef[qid])
    #print(norm(kernel_alpha[qid] @ data_alpha[nns[qid]]))
    #print(norm(kernel_alpha[qid] @ data_alpha[2343]))
    print(distance_func(kernel_alpha[qid], data_alpha[nns[qid]].reshape((1, dimensions[dataset]+2)), c_coef[qid]))
    print(distance_func(kernel_alpha[qid], data_alpha[2343].reshape((1, dimensions[dataset]+2)), c_coef[qid]))    
    print(distance_func(kernel_alpha[qid], data_alpha[5532].reshape((1, dimensions[dataset]+2)), c_coef[qid]))
    print(distance_func(kernel_alpha[qid], data_alpha[538].reshape((1, dimensions[dataset]+2)), c_coef[qid]))

run2("audio")