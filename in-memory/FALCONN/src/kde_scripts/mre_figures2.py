import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fvecs import *
import sys, os

datasets = ["deep", "enron", "trevi", "glove", "sift"]
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
    
    with open(os.path.join(exp_path, dataset, "ann_ps3.json")) as fin:
        conf = json.load(fin)

    
    #ax.plot(x_bins, mres[2], label="TRI")

    mp_tuple, mp_recalls = tuple(fvecs_read_non_uniform(conf["mp prob filename"]))
    l = conf["hash table parameters"][1]["l"]
    bucket_width = conf["hash table parameters"][1]["bucket width"]
    _, _, mpstep = mp_tuple
    mp_recalls = 1.0 - (1.0 - mp_recalls) ** l
    mp_size = len(mp_recalls)

    def mp_prob(distance):
        idx = int(distance / bucket_width / mpstep)
        if idx < 0:
            ids = 0
        if idx >= mp_size:
            idx = mp_size - 1
        return mp_recalls[idx]

    def get_curve(filename):
        result_path = os.path.join(exp_path, dataset, filename)
        with open(result_path) as fin:
            x_bins = [float(x) for x in fin.readline().split()]
            mres = [float(x) for x in fin.readline().split()]
        recalls = [mp_prob(x) for x in x_bins]
        return recalls, mres

    
    plt.clf()
    ax = plt.gca()
    plt.xscale('log')
    rec, mre = get_curve("mle_mres.txt")
    ax.plot(rec, mre, label="basic")
    rec, mre = get_curve("tbl_mres.txt")
    ax.plot(rec, mre, label="alternative")

    ax.set_xlabel("MP Recall")
    ax.set_ylabel("Mean Relative Error")
    ax.legend(loc="upper right")
    plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/recall_mre_{}.pdf".format(
            dataset))
