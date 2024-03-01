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
query = 0
if "-q" in sys.argv:
    idx = sys.argv.index("-q")
    query = int(sys.argv[idx+1])
distance = fvecs_read(config["distance file"])
num_experiments = 1000
if "-e" in sys.argv:
    idx = sys.argv.index("-e")
    num_experiments = int(sys.argv[idx+1])
lb = 7
if "-lb" in sys.argv:
    idx = sys.argv.index("-lb")
    lb = int(sys.argv[idx+1])
ub = 14
if "-ub" in sys.argv:
    idx = sys.argv.index("-ub")
    ub = int(sys.argv[idx+1])
PLOT_RANGE = [lb, ub]

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
for i in range(l_max):
    candidates.append([])
pointer = 5
l = config["hash table parameters"][0]["l"]
l = l_val if l_val is not None else l
for cand in range(length):
    id, table, bucket = struct.unpack("@iii", candfile[pointer: pointer + 12])
    pointer += 12
    candidates[table].append(id)

gt_histogram, edges = np.histogram(distance[query, :], bins=50, range=PLOT_RANGE)
rng = np.random.default_rng()

mp_recalls = fvecs_read_non_uniform(config["mp prob filename"])[1]
mp_recalls = 1.0 - (1.0 - mp_recalls) ** l
bucket_width = config["hash table parameters"][0]["bucket width"]

def mp_prob(distance):
    idx = int(distance / bucket_width * 10000)
    if idx < 0:
        ids = 0
    if idx >= 30000:
        idx = 29999
    return mp_recalls[idx]

def rel_errors():
    tables = rng.choice(l_max, l, replace=False)
    candidates_flat = []
    for tt in tables:
        candidates_flat.extend(candidates[tt])
    candidates_flat = set(candidates_flat)
    weights = []
    cand_dist  = []
    for cand in candidates_flat:
        cand_dist.append(distance[query, cand])
        weights.append(mp_prob(distance[query, cand]))
    weights = 1.0 / np.array(weights)

    estimated_hist, _ = np.histogram(cand_dist, bins=50, range=PLOT_RANGE, weights=weights)
    diff = np.abs(estimated_hist - gt_histogram, dtype=np.float64)
    return np.divide(diff, gt_histogram, out=np.zeros_like(diff), where=gt_histogram!=0)

mre = np.zeros_like(gt_histogram, dtype=np.float64)
for exp in range(num_experiments):
    mre += rel_errors()
mre /= num_experiments
x_vals = [(edges[i+1] + edges[i]) / 2 for i in range(len(edges)-1)]
matplotlib.rcParams.update({'font.size': 12})
plt.plot(x_vals, mre)
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('Mean Relative Error')

exp_path = pathlib.Path(sys.argv[1]).parent.parent 
file_path = exp_path / "plots" / "mre-q{}-l{}.pdf".format(query, l)
plt.savefig(str(file_path))
