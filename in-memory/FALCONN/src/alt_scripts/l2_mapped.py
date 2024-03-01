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

universe["audio"] = 100000

def objective_fun(u, alpha, mm, mq, q):
    return (u**2 + alpha**2) ** 2 * (mm **2 + 2* mq ** 2 / alpha**2 + q ** 4 / alpha ** 4)

def optimize_alpha(kernel, query, univ):
    mm = np.linalg.norm(kernel.T @ kernel)
    mq = np.linalg.norm(kernel.T @ query)
    q = np.linalg.norm(query)
    result = optimize.minimize_scalar(lambda alpha: objective_fun(univ, alpha, mm, mq, q))
    return result.x

def compute_beta(u, alpha, kernel, query):
    mm = np.linalg.norm(kernel.T @ kernel)
    mq = np.linalg.norm(kernel.T @ query)
    q = np.linalg.norm(query)
    return sqrt(1+(u/alpha)**2) / (mm **2 + 2* mq ** 2 / alpha**2 + q ** 4 / alpha ** 4) ** (1/4)

def compute_beta_var(u, alpha, kernel):
    return sqrt(1+(u/alpha)**2) / np.linalg.norm(kernel.T @ kernel) ** (1/2)

def compute_distance(u, alpha, beta, kernel, query, nn_x, spotaneous):
    mm = np.linalg.norm(kernel.T @ kernel)
    mq = np.linalg.norm(kernel.T @ query)
    q = np.linalg.norm(query)
    mx = np.linalg.norm(kernel @ nn_x - query)
    print(((u/alpha) ** 2 + 1) ** 2, beta ** 4 * (mm **2 + 2* mq ** 2 / alpha**2 + q ** 4 / alpha ** 4),
     2 * beta ** 2 * mx ** 2 / alpha **2 , 2 * beta ** 2 * np.linalg.norm(kernel @ spotaneous - query) ** 2 / alpha **2)
    distance = sqrt(((u/alpha) ** 2 + 1) ** 2 + \
        beta ** 4 * (mm **2 + 2* mq ** 2 / alpha**2 + q ** 4 / alpha ** 4) + \
            2 * beta ** 2 * mx ** 2 / alpha **2)
    #print(mm **2, 2* mq ** 2 / alpha**2, q ** 4 / alpha ** 4)
    return distance, 2 * beta ** 2 * np.linalg.norm(kernel @ spotaneous - query) ** 2 / alpha **2 / distance ** 2


def compute_distance_var(u, alpha, beta, kernel, nn_x):
    mx = np.linalg.norm(kernel @ nn_x)
    distance = sqrt(((u/alpha) ** 2 + 1) ** 2 + \
        beta ** 4 * np.linalg.norm(kernel.T @ kernel) ** 2 + \
            2 * beta ** 2 * mx ** 2)
    #print(mm **2, 2* mq ** 2 / alpha**2, q ** 4 / alpha ** 4)
    return distance

def run(dataset):
    univ = universe[dataset] + 1 # consider the added 1 dimension.
    gt_file = "ground_truth/" + dataset + ".txt"
    gt_file2 = "ground_truth/" + dataset + "2.txt"
    nns = np.empty(1000, dtype=np.int32)
    alphas = np.empty(1000)
    betas = np.empty(1000)
    dists = np.empty(1000)
    ratios = np.empty(1000)
    
    with open(gt_file, "r") as gtin:
        for i, line in enumerate(gtin.readlines()):
            nns[i] = int(line.split()[0])

    with open(gt_file2, "r") as gtin:
        for i, line in enumerate(gtin.readlines()):
            nns[i+100] = int(line.split()[0])


    kernel_file = "half_kernels/" + dataset + ".fvecs"
    kernels = fvecs_read(kernel_file)
    kernels = kernels.reshape((1000, dimensions[dataset], dimensions[dataset]))

    query_file = "dataset/" + dataset + "-testC.fvecs"

    raw_file = "dataset/" + dataset + "-trainC.fvecs"
    data = fvecs_read(raw_file)

    queries = fvecs_read(query_file)

    for i in range(1000):
        alphas[i] = optimize_alpha(kernels[i], kernels[i]@ queries[i].T, universe[dataset])
    
    alpha_mean = alphas.mean()
    alpha_std = alphas.std()
    alpha_rot = universe[dataset] * dimensions[dataset] ** (-1/6)

    data_alpha = np.empty((data.shape[0], dimensions[dataset] + 1))
    data_alpha[:, :-1] = data / alpha_rot
    data_alpha[:, -1] = -1
    kernel_alpha = np.empty((1000, dimensions[dataset], dimensions[dataset] + 1))
    for i in range(1000):
        kernel_alpha[i, :, :-1] = kernels[i]
        kernel_alpha[i, :, -1] = kernels[i] @ queries[i].T / alpha_rot


    for i in range(1000):
        betas[i] = compute_beta_var(universe[dataset], alpha_rot, kernel_alpha[i])

    beta_mean = betas.mean()
    beta_std = betas.std()

    for i in range(1000):
        dists[i] = compute_distance_var(universe[dataset], alpha_rot, betas[i], 
        kernel_alpha[i], data_alpha[nns[i]].T)
        ratios[i] =  compute_distance_var(universe[dataset], alpha_rot, betas[i], 
        kernel_alpha[i], data_alpha[10000].T) / dists[i] -1


    dist_mean = dists.mean()
    dist_std = dists.std()
    ratio_mean = ratios.mean()

    return alpha_mean, alpha_std, alpha_rot, beta_mean, beta_std, dist_mean, dist_std, ratio_mean

    
#for dataset in datasets:
#    run(dataset)
if __name__ == "__main__":
    print(run("audio"))
