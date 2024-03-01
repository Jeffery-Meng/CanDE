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

lb, ub, step = tuple(config["histogram bins"])
bins = np.arange(lb, ub, step)

candidate_file = config["candidate filename"] + "_q_{}.bin".format(query)
with open(candidate_file, "rb") as fin:
    candfile = fin.read()

dataset_name = pathlib.PurePath(sys.argv[1]).parent.name

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

candidates = set(candidates)
print(len(candidates))
bucket_width = config["hash table parameters"][0]["bucket width"]

mp_tuple, mp_recalls = tuple(fvecs_read_non_uniform(config["mp prob filename"]))
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

weights = []
cand_dist  = []
for cand in candidates:
    cand_dist.append(distance[query, cand])
    weights.append(mp_prob(distance[query, cand]))
weights = 1.0 / np.array(weights)

dists = list(bins)
probs = []
for i in dists:
    probs.append( mp_prob(i))
matplotlib.rcParams.update({'font.size': 12})
plt.hist(distance[query, :], range=(lb,ub), bins=bins, alpha=0.7, label="ground truth")
plt.hist(cand_dist, bins=bins, range=(lb, ub), weights=weights, alpha=0.7, color="r", label="estimated")
print(np.sum(weights))
plt.legend(loc="upper right")
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('# of points')
ax2 = ax1.twinx()
ax2.plot(dists, probs)
ax2.set_ylabel('recall')

exp_path = pathlib.Path(sys.argv[1]).parent.parent 
file_path = exp_path / "plots" / "{}-mleind-q{}-l{}.pdf".format(dataset_name, query, l)
plt.savefig(str(file_path))
