import subprocess
import pathlib
import os, sys
import json
from copy import deepcopy
import numpy as np
from fvecs import *
import matplotlib.pyplot as plt

exp_path = sys.argv[1]

num_bins = 50
if "--bins" in sys.argv:
    idx = sys.argv.index("--bins")
    num_bins = int(sys.argv[idx+1])

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

config_file = os.path.join(exp_path, "ann_ps3.json")
with open(config_file) as fin:
    config = json.load(fin)

distance = fvecs_read(config["distance file"])
lb = np.quantile(distance, 0.01)
ub = np.quantile(distance, 0.99)
ub = 2.5 * lb if ub > 2.5 * lb else ub
step = (ub - lb) / num_bins
config["histogram bins"] = [lb, ub, step]
config["result filename"] = os.path.join(exp_path, "mle_times.txt")
config["row id filename"] = os.path.join(exp_path, "mle_mres.txt")

mle_config_path = os.path.join(exp_path, "ann_mle.json")
with open(mle_config_path, "w") as fout:
    json.dump(config, fout, indent=4)

subprocess.run([os.path.join(bin_path, "falconn_disthist_mle"),
    mle_config_path])

config["result filename"] = os.path.join(exp_path, "tbl_times.txt")
config["row id filename"] = os.path.join(exp_path, "tbl_mres.txt")
tbl_config_path = os.path.join(exp_path, "ann_tbl.json")
with open(tbl_config_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_disthist_tblwise"),
    tbl_config_path])

config["result filename"] = os.path.join(exp_path, "gamma_mres.txt")
gamma_config_path = os.path.join(exp_path, "ann_gamma.json")
with open(gamma_config_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run(["python3", os.path.join(this_path, "gamma.py"),
    exp_path])

config["result filename"] = os.path.join(exp_path, "knn_times.txt")
config["query mode"] = "knn time"
del config["hash table parameters"][1]
knn_config_path = os.path.join(exp_path, "ann_time.json")
with open(knn_config_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn"), "-cf",
    knn_config_path])

def read_data(filename):
    with open(filename) as fin:
        a = [float(x) for x in fin.readline().split()]
        b = [float(x) for x in fin.readline().split()]
    return a, b

plt.clf()
ax = plt.gca()
xbin, histogram = read_data(os.path.join(exp_path, "mle_mres.txt"))
ax.plot(xbin, histogram, label="basic")
xbin, histogram = read_data(os.path.join(exp_path, "tbl_mres.txt"))
ax.plot(xbin, histogram, label="alternative")
xbin, histogram = read_data(os.path.join(exp_path, "gamma_mres.txt"))
ax.plot(xbin, histogram, label="baseline")
ax.set_ylim((0, 0.8))
#ax.plot(x_bins, mres[2], label="TRI")
ax.set_xlabel("distance")
ax.set_ylabel("Mean Relative Error")
ax.legend(loc="upper right")

exp_split = exp_path.split("/")
dataset = exp_split[-1] if exp_split[-1] else exp_split[-2]
plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/{}_mres.pdf".format(
        dataset))