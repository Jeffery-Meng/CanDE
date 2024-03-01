import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fvecs import *
import sys, os

datasets = ["deep", "gist", "random"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"

def get_mre(result_path, scheme):
    mre_total = None
    for qid in range(1000):
        filename = "{}-mre-q{}.txt".format(scheme, qid)
        filename = os.path.join(result_path, filename)
        mre = np.loadtxt(filename)
        if mre_total is None:
            mre_total = mre
        else:
            mre_total += mre
    return mre_total / 1000


for dataset in datasets:
    result_path = os.path.join(exp_path, dataset, "summary")
    mres = []
    for scheme in ["mle", "combined", "gamma"]:
        mres.append(get_mre(result_path, scheme))
    
    with open(os.path.join(exp_path, dataset, "ann_de.json")) as fin:
        conf = json.load(fin)
    x_bins = np.arange(*conf["histogram bins"])[:-1]

    plt.clf()
    ax = plt.gca()
    ax.plot(x_bins, mres[0], label="basic")
    ax.plot(x_bins, mres[1], label="alternative")
    ax.plot(x_bins, mres[2], label="gamma fit")
    ax.set_ylim((0, 0.8))
    #ax.plot(x_bins, mres[2], label="TRI")
    ax.set_xlabel("distance")
    ax.set_ylabel("Mean Relative Error")
    ax.legend(loc="upper right")
    plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plots/final2_{}.pdf".format(
            dataset))
