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
query = 0
if "-q" in sys.argv:
    idx = sys.argv.index("-q")
    query = int(sys.argv[idx+1])
distance = fvecs_read(config["distance file"])
nq, n = distance.shape

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

# deliberately tune down l
l = int(l * 0.9)
mp_recalls = fvecs_read_non_uniform(config["mp prob filename"])[1]
mp_recalls_table = []
for i in range(2 * l):
    mp_recalls_table.append( 1.0 - (1.0 - mp_recalls) ** i)

def mp_prob(distance, i):
    idx = int(distance / bucket_width * 10000)
    if idx < 0:
        ids = 0
    if idx >= 30000:
        idx = 29999
    return mp_recalls_table[i][idx]


candidate_distances = distance[query, candidates]
candidate_bins, edges = np.histogram(candidate_distances, bins=50, range=PLOT_RANGE)
x_vals = [(edges[i+1] + edges[i]) / 2 for i in range(len(edges)-1)]
probs = np.zeros_like(candidate_bins, dtype=np.float64)

for ll in range(2 * l -1, 0, -1):
    sum = 0
    for i in range(len(probs)):
        probs[i] = mp_prob(x_vals[i], ll)
        sum += candidate_bins[i] / probs[i]
    if sum >= n:
        break
    
last_idx = candidate_bins.shape - np.argmax((candidate_bins != 0)[::-1]) - 1
gamma_max = 1.0 / (1.0 - probs[last_idx]) - 1e-8

def mle_func(gamma):
    sum = 0
    for i in range(len(candidate_bins)):
        sum += candidate_bins[i] / (1.0 - gamma + gamma * probs[i])
    return sum - n

gamma = bisect(mle_func, 0, gamma_max)
expected_bins = candidate_bins / (1.0 - gamma + gamma * probs)
print(ll, gamma)

matplotlib.rcParams.update({'font.size': 12})
plt.hist(distance[query, :], range=PLOT_RANGE, bins=50, alpha=0.7, label="ground truth")
plt.stairs(expected_bins, edges, alpha=0.7, color="r", label="estimated", linewidth=3)
plt.legend(loc="upper right")
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('# of points')


exp_path = pathlib.Path(sys.argv[1]).parent.parent 
file_path = exp_path / "plots" / "mnist-mle2-q{}-l{}.pdf".format(query, ll)
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