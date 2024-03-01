import sys
import json, os.path, struct
from fvecs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib

with open(sys.argv[1]) as fin:
    config = json.load(fin)

query = 0
if "-q" in sys.argv:
    idx = sys.argv.index("-q")
    query = int(sys.argv[idx+1])
distance = fvecs_read(config["distance file"])
qn, n = distance.shape

candidate_file = config["candidate filename"] + "_q_{}.bin".format(query)
with open(candidate_file, "rb") as fin:
    candfile = fin.read()

mode = struct.unpack("@B", candfile[:1])
mode = mode[0]
assert(mode == 0b11100000)
length = struct.unpack("@i", candfile[1:5])
length = length[0]

pointer = 5
l_max = config["hash table parameters"][0]["l"]
candidates = []
for i in range(l_max):
    candidates.append([])
for cand in range(length):
    id, table, bucket = struct.unpack("@iii", candfile[pointer: pointer + 12])
    pointer += 12
    candidates[table].append(id)

candidates_flat = []
for cand in candidates:
    candidates_flat.extend(cand)
candidates_flat.sort()
candidates_unique = set(candidates_flat)
candidate_counts = dict.fromkeys(candidates_unique, 0)
for cand in candidates_flat:
    candidate_counts[cand] += 1

mp_recalls = fvecs_read_non_uniform(config["mp prob filename"])[1]
mp_recalls *= l_max
bucket_width = config["hash table parameters"][0]["bucket width"]

def mp_prob(distance):
    idx = int(distance / bucket_width * 10000)
    if idx < 0:
        ids = 0
    if idx >= 30000:
        idx = 29999
    return mp_recalls[idx]
dists = np.arange(5, 15, 0.1)
probs = []
keys = []
for dd in dists:
    probs.append( mp_prob(dd))
    keys.append(f'{dd:.1f}')

bins = dict.fromkeys(keys, 0)
counts = dict.fromkeys(keys, 0)
for id in range(n):
    cur_distance = distance[query, id]
    if cur_distance >= 5 and cur_distance < 15:
        key = f'{cur_distance:.1f}'
        bins[key] += candidate_counts.get(id, 0)
        counts[key] += 1

recall_est = []
for dd in dists:
    if counts[f'{dd:.1f}'] > 0:
        recall_est.append(bins[f'{dd:.1f}'] / counts[f'{dd:.1f}'])
    else:
        recall_est.append(0)

plt.plot(dists, probs)
plt.plot(dists, recall_est)
plt.xlim(8, 15)
plt.ylim(0, 20)
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('Average # of occurrences (among 500 tables)')
exp_path = pathlib.Path(sys.argv[1]).parent.parent 
file_path = exp_path / "plots" / "candi-q{}.pdf".format(query)
plt.savefig(str(file_path))
