import numpy as np
import matplotlib.pyplot as plt
from fvecs import *
from sys import argv

datasets = ["audio", "mnist", "enron", "trevi", "glove", "gist", "deep", "sift"]

def run(dataset):
    path = "dataset/" + dataset + "-train.fvecs"
    data = fvecs_read(path)
    l2norms = np.linalg.norm(data, axis=1)

    print(dataset, np.min(l2norms), np.max(l2norms))
    nbins = int(data.shape[0] ** (1/3))
    plt.hist(l2norms, nbins)
    plt.savefig("l2hist/" + dataset + ".png", dpi=600)
    plt.clf()

#for dataset in datasets:
#    run(dataset)

def run_enron():
    dataset = "enron"
    cut_off = 1000
    path = "dataset/" + dataset + "-train.fvecs"
    data = fvecs_read(path)
    l2norms = np.linalg.norm(data, axis=1)

    small_norms = l2norms[l2norms < cut_off]

    print(dataset, np.min(l2norms), np.max(l2norms))
    nbins = int(data.shape[0] ** (1/3))
    plt.hist(small_norms, nbins)
    plt.savefig("l2hist/enron2.png", dpi=600)
    plt.clf()

def run_c(dataset):
    path = "dataset/" + dataset + "-trainC.fvecs"
    data = fvecs_read(path)
    l2norms = np.linalg.norm(data, axis=1)

    print(dataset, np.min(l2norms), np.max(l2norms))
    nbins = int(data.shape[0] ** (1/3))
    plt.hist(l2norms, nbins)
    plt.savefig("l2hist/" + dataset + "C.png", dpi=600)
    plt.clf()

if __name__ == "__main__":
    if len(argv) > 1:
        run(argv[1])
