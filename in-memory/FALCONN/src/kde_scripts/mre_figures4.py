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
    with open(filename) as fin:
        a = [float(x) for x in fin.readline().split()]
        b = [float(x) for x in fin.readline().split()]
    return a, b

plt.clf()

f = plt.figure()
ax = plt.gca()
params = {'legend.fontsize': 22,
         'axes.labelsize': 22,
         'axes.titlesize':22,
         'xtick.labelsize':22,
         'ytick.labelsize':22}

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,3))
xbin1, histogram1 = read_data(os.path.join(exp_path, "mle_mres.txt"))
ax.plot(xbin1, histogram1, "-", linewidth=3, label="basic")
xbin2, histogram2 = read_data(os.path.join(exp_path, "tbl_mres.txt"))
ax.plot(xbin2, histogram2, "--", linewidth=3,  label="alternative")
xbin3, histogram3 = read_data(os.path.join(exp_path, "gamma_mres.txt"))
ax.plot(xbin3, histogram3, ":", linewidth=3, label="baseline")
ax.set_ylim((0, 0.8))
#ax.plot(x_bins, mres[2], label="TRI")
ax.set_xlabel("Q2D distance",fontsize=22)
ax.set_ylabel("MRE",fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
# ax.legend(loc="upper right")

exp_split = exp_path.split("/")
dataset = exp_split[-1] if exp_split[-1] else exp_split[-2]
print(dataset)
if dataset == "gist":
    print(xbin1, histogram1)
plt.rcParams.update(params)
plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/{}_mres.pdf".format(
        dataset), bbox_inches="tight")
fig = pylab.figure()
figlegend = pylab.figure(figsize=(3,2))
ax = fig.add_subplot(111)
lines = []
lines = ax.plot(xbin1, histogram1, "-", xbin2, histogram2, "--", xbin3, histogram3, ":", linewidth=3)
figlegend.legend(lines, ('basic', 'alternative', "baseline"), 'center')
figlegend.savefig('/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/legend.pdf')


