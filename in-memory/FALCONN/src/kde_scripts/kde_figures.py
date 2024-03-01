import subprocess
import pathlib
import os, sys
import pylab
import json
from copy import deepcopy
import numpy as np
from fvecs import *
import matplotlib.pyplot as plt
import matplotlib

exp_path = sys.argv[1]

def read_data(filename):
    a = []
    b = []
    c = []
    with open(filename) as fin:
        for line in fin.readlines():
            vec = [float(x) for x in line.split()]
            a.append(vec[0])
            b.append(vec[1])
            c.append(vec[2])
    return a, b, c

def get_best_param(path, prefix):
    min_val = 100
    min_a, min_b = (None, None)
    for file in os.listdir(path):
        if prefix not in file:
            continue
        a, b, c = read_data(os.path.join(path, file))
        b_x = b[c==1000]
        if np.min(b_x) < min_val:
            min_val = np.min(b_x)
            min_a, min_b = (a, b)
    return min_a, min_b

plt.clf()

f = plt.figure()
ax = plt.gca()
params = {'legend.fontsize': 22,
         'axes.labelsize': 22,
         'axes.titlesize':22,
         'xtick.labelsize':22,
         'ytick.labelsize':22}

config_file = os.path.join(exp_path, "kde.json")
with open(config_file) as fin:
    config = json.load(fin)         
from scott_rot import scott_rot
scott = scott_rot(config["data filename"])[-1]

xbin1, histogram1, _ = read_data(os.path.join(exp_path, "kde_mle.txt"))
ax.plot(xbin1, histogram1, "-", linewidth=3, label="MLE")
xbin2, histogram2 = get_best_param(os.path.join(exp_path, "summary"), "kde_tbl_mres")
ax.plot(xbin2, histogram2, "--", linewidth=3,  label="TBL1")
xbin3, histogram3 = get_best_param(os.path.join(exp_path, "summary"), "kde_v3_mres")
ax.plot(xbin3, histogram3, ":", linewidth=3, label="TBL2")
plt.axvline(x = scott, linestyle=":", color="grey", linewidth=1, label = 'Scott')
ax.set_ylim((0, 0.2))
plt.xscale("log")
#ax.plot(x_bins, mres[2], label="TRI")
ax.set_xlabel("Gamma (Locality)",fontsize=22)
ax.set_ylabel("MRE",fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
# ax.legend(loc="upper right")

exp_split = exp_path.split("/")
dataset = exp_split[-1] if exp_split[-1] else exp_split[-2]
print(dataset)
plt.rcParams.update(params)
plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/{}_KDE.pdf".format(
        dataset), bbox_inches="tight")
fig = pylab.figure()
figlegend = pylab.figure(figsize=(3,2))
ax = fig.add_subplot(111)
lines = []
lines = ax.plot(xbin1, histogram1, "-", xbin2, histogram2, "--", xbin3, histogram3, ":", linewidth=3)
lines.extend(ax.plot(xbin1, histogram1, ":",color="grey", linewidth=1))
figlegend.legend(lines, ('CANDE-P', 'CANDE-I', "CANDE-I/ALT", "Scott"), 'center')
figlegend.savefig('/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/legend.pdf')


