import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fvecs import *
import sys, os

datasets = ["deep", "enron", "trevi", "glove", "sift", "audio", "mnist"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"

def get_mre(result_path, prefix):
    mre_total = None
    for qid in range(1000):
        filename = "{}-q{}.txt".format(prefix, qid)
        filename = os.path.join(result_path, filename)
        mre = np.loadtxt(filename)
        if mre_total is None:
            mre_total = mre
        else:
            mre_total += mre
    return mre_total / 1000


for dataset in datasets:
    
    with open(os.path.join(exp_path, dataset, "ann_mle.json")) as fin:
        conf = json.load(fin)

    def get_curve(filename):
        result_path = os.path.join(exp_path, dataset, filename)
        with open(result_path) as fin:
            x_bins = [float(x) for x in fin.readline().split()]
            mres = [float(x) for x in fin.readline().split()]
        return x_bins, mres
    def get_line(filename):
        result_path = os.path.join(exp_path, dataset, filename)
        with open(result_path) as fin:
            x_bins = [float(x) for x in fin.readline().split()]
        return x_bins
    #ax.plot(x_bins, mres[2], label="TRI")
    cand_num = get_line("knn_candidate_count.txt")
    plt.clf()
    ax = plt.gca()
    plt.xscale('log') 
    
    rec, mre = get_curve("mle_mres.txt")
    ax.plot(cand_num, mre, label="basic")
    rec, mre = get_curve("tbl_mres.txt")
    ax.plot(cand_num, mre, label="alternative")

    ax.set_xlabel("Average Number of Effective Candidates")
    ax.set_ylabel("Mean Relative Error")
    ax.legend(loc="upper right")
    plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/candN_mre_{}.pdf".format(
            dataset))
