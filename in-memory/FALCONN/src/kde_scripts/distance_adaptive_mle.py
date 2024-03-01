import sys
import json, os.path, struct
from fvecs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import bisect
import pathlib

with open(sys.argv[1]) as fin:
    config = json.load(fin)

l_val = None
if "-l" in sys.argv:
    idx = sys.argv.index("-l")
    l_val = int(sys.argv[idx+1])

distance = fvecs_read(config["distance file"])
nq, n = distance.shape

query = 0
if "-q" in sys.argv:
    idx = sys.argv.index("-q")
    query = int(sys.argv[idx+1])
lb = 7
if "-lb" in sys.argv:
    idx = sys.argv.index("-lb")
    lb = int(sys.argv[idx+1])
ub = 14
if "-ub" in sys.argv:
    idx = sys.argv.index("-ub")
    ub = int(sys.argv[idx+1])
PLOT_RANGE = [lb, ub]
pthre = 0.05
if "-pt" in sys.argv:
    idx = sys.argv.index("-pt")
    pthre = float(sys.argv[idx+1])

mre = None
for query in range(query, nq):
    candidate_file = config["candidate filename"] + "_q_{}.bin".format(query)
    with open(candidate_file, "rb") as fin:
        candfile = fin.read()

    mode = struct.unpack("@B", candfile[:1])
    mode = mode[0]
    assert(mode == 0b11100000)
    length = struct.unpack("@i", candfile[1:5])
    length = length[0]
    candidates = []
    pointer = 5
    l = config["hash table parameters"][0]["l"]
    l = l_val if l_val is not None else l
    for cand in range(length):
        id, table, bucket = struct.unpack("@iii", candfile[pointer: pointer + 12])
        pointer += 12
        if table < l  and table >= 0:
            candidates.append(id)

    candidates = list(set(candidates))
    bucket_width = config["hash table parameters"][0]["bucket width"]

    mp_recalls = fvecs_read_non_uniform(config["mp prob filename"])[1]
    mp_recalls = 1.0 - (1.0 - mp_recalls) ** l

    def mp_prob(distance):
        idx = int(distance / bucket_width * 10000)
        if idx < 0:
            ids = 0
        if idx >= 30000:
            idx = 29999
        return mp_recalls[idx]


    candidate_distances = distance[query, candidates]
    candidate_bins, edges = np.histogram(candidate_distances, bins=50, range=PLOT_RANGE)
    x_vals = [(edges[i+1] + edges[i]) / 2 for i in range(len(edges)-1)]
    probs = np.zeros_like(candidate_bins, dtype=np.float64)
    for i in range(len(probs)):
        probs[i] = mp_prob(x_vals[i])
    gamma_max = np.min(probs[probs>pthre])

    def mle_func(gamma):
        sum = 0
        for i in range(len(candidate_bins)):
            if probs[i] > pthre:
                sum += candidate_bins[i] / (1.0 - gamma + gamma * probs[i])
            else:
                sum += candidate_bins[i] / probs[i]
        return sum - n

    gamma_max = 1.0 / (1.0 - gamma_max) - 1e-8
    if mle_func(0) * mle_func(gamma_max) > 0:
        expected_bins = candidate_bins / probs
        scale_ratio = n / np.sum(expected_bins)
        expected_bins = expected_bins * scale_ratio
        print(query, "scale", scale_ratio)
    else:
        gamma = bisect(mle_func, 0, gamma_max)
        expected_bins = np.zeros_like(candidate_bins)
        for i in range(len(candidate_bins)):
            if probs[i] > pthre:
                expected_bins[i] = candidate_bins[i] / (1.0 - gamma + gamma * probs[i])
            else:
                expected_bins[i] = candidate_bins[i] / probs[i]
        print(query, gamma)
    
    plt.clf()
    matplotlib.rcParams.update({'font.size': 12})
    plt.hist(distance[query, :], range=PLOT_RANGE, bins=50, alpha=0.7, label="ground truth")
    plt.stairs(expected_bins, edges, alpha=0.7, color="r", label="estimated", linewidth=3)
    plt.legend(loc="upper right")
    ax1 = plt.gca()
    ax1.set_xlabel('distance')
    ax1.set_ylabel('# of points')


    exp_path = pathlib.Path(sys.argv[1]).parent.parent 
    file_path = exp_path / "plots" / "mnist-amle2-q{}-l{}.pdf".format(query, l)
    plt.savefig(str(file_path))

    gt_histogram, edges = np.histogram(distance[query, :], bins=50, range=PLOT_RANGE)
    diff = np.abs(expected_bins - gt_histogram, dtype=np.float64)
    if mre is None:
        mre = np.zeros_like(diff)
    mre += np.divide(diff, gt_histogram, out=np.zeros_like(diff), where=gt_histogram!=0)

mre /= nq
plt.clf()
x_vals = [(edges[i+1] + edges[i]) / 2 for i in range(len(edges)-1)]
matplotlib.rcParams.update({'font.size': 12})
plt.plot(x_vals, mre)
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('Mean Relative Error')


exp_path = pathlib.Path(sys.argv[1]).parent.parent 
file_path = exp_path / "plots" / "amle2-mre.pdf"
file_path2 = exp_path / "mnist" / "amle2-mre.txt"
np.savetxt(file_path2, mre)
plt.savefig(str(file_path))

    # Print result
    # exp_path = pathlib.Path(sys.argv[1]).parent.parent 
    # file_path = exp_path / "mnist-estimated-q{}-l{}-result.txt".format(query, l)
    # file_path_temp = exp_path / "ground_truth-q{}.txt".format(query)
    # arr_temp = np.full((edges.shape[0],2),-1.0)
    # arr_temp[1:,0]=expected_bins[:]
    # arr_temp[:,1]=edges[:]
    # np.savetxt(file_path_temp,distance[query, :])
    # np.savetxt(file_path,arr_temp)