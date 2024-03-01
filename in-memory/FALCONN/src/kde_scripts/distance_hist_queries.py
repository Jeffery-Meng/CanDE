import sys
import json, os.path, struct
from fvecs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib

with open(sys.argv[1]) as fin:
    config = json.load(fin)

l_val = None
if "-l" in sys.argv:
    idx = sys.argv.index("-l")
    l_val = int(sys.argv[idx+1])

distance = fvecs_read(config["distance file"])
qn, n = distance.shape
lb = np.min(distance)
if "-lb" in sys.argv:
    idx = sys.argv.index("-lb")
    lb = float(sys.argv[idx+1])
ub = np.max(distance)
if "-ub" in sys.argv:
    idx = sys.argv.index("-ub")
    ub = float(sys.argv[idx+1])
PLOT_RANGE = [lb, ub]

l = config["hash table parameters"][0]["l"]
l = l_val if l_val is not None else l

rng = np.random.default_rng()

mp_tuple, mp_recalls = tuple(fvecs_read_non_uniform(config["mp prob filename"]))
_, _, mpstep = mp_tuple
mp_recalls = 1.0 - (1.0 - mp_recalls) ** l
mp_size = len(mp_recalls)
bucket_width = config["hash table parameters"][0]["bucket width"]
exp_path = pathlib.Path(sys.argv[1]).parent.parent
dataset_name = pathlib.PurePath(sys.argv[1]).parent.name

def mp_prob(distance):
    idx = int(distance / bucket_width / mpstep)
    if idx < 0:
        ids = 0
    if idx >= mp_size:
        idx = mp_size - 1
    return mp_recalls[idx]

def rel_errors(query):
    print(query)
    candidate_file = config["candidate filename"] + "_q_{}.bin".format(query)
    with open(candidate_file, "rb") as fin:
        candfile = fin.read()
    mode = struct.unpack("@B", candfile[:1])
    mode = mode[0]
    assert(mode == 0b11100000)
    length = struct.unpack("@i", candfile[1:5])
    length = length[0]
    candidates = []
    l_max = config["hash table parameters"][0]["l"]
    pointer = 5
    for cand in range(length):
        id, table, bucket = struct.unpack("@iii", candfile[pointer: pointer + 12])
        pointer += 12
        if table < l:
            candidates.append(id)
    candidates = set(candidates)
    weights = []
    cand_dist  = []
    for cand in candidates:
        cand_dist.append(distance[query, cand])
        weights.append(mp_prob(distance[query, cand]))
    weights = 1.0 / np.array(weights)

    gt_histogram, edges = np.histogram(distance[query, :], bins=50, range=PLOT_RANGE)
    estimated_hist, _ = np.histogram(cand_dist, bins=50, range=PLOT_RANGE, weights=weights)

    plt.clf()
    matplotlib.rcParams.update({'font.size': 12})
    plt.hist(distance[query, :], range=PLOT_RANGE, bins=50, alpha=1, label="ground truth")
    plt.stairs(estimated_hist, edges, alpha=0.7, color="r", label="estimated", linewidth=3)
    plt.legend(loc="upper right")
    ax1 = plt.gca()
    ax1.set_xlabel('distance')
    ax1.set_ylabel('# of points')
     
    file_path = exp_path / "plots" / "{}-mle-q{}-l{}.pdf".format(dataset_name, query, l)
    plt.savefig(str(file_path))

    diff = np.abs(estimated_hist - gt_histogram, dtype=np.float64)
    return np.divide(diff, gt_histogram, out=np.zeros_like(diff), where=gt_histogram!=0), edges

mre, edges = rel_errors(0)
for qid in range(1, qn):
    mre += rel_errors(qid)[0]
mre /= qn
x_vals = [(edges[i+1] + edges[i]) / 2 for i in range(len(edges)-1)]
matplotlib.rcParams.update({'font.size': 12})
plt.clf()
plt.plot(x_vals, mre)
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('Mean Relative Error')

exp_path = pathlib.Path(sys.argv[1]).parent.parent 
file_path = exp_path / "plots" / "{}-mre-allq-l{}.pdf".format(dataset_name, l)
plt.savefig(str(file_path))

file_path2 = exp_path / dataset_name / "mle-mre.txt"
np.savetxt(file_path2, mre)