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
distance = distance[query, :]

lb, ub, step = tuple(config["histogram bins"])
bins = np.arange(lb, ub, step)
np.seterr(all="ignore")
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
l = config["hash table parameters"][1]["l"]
l = l_val if l_val is not None else l
for cand in range(length):
    id, table, bucket = struct.unpack("@iii", candfile[pointer: pointer + 12])
    pointer += 12
    if table < l  and table >= 0:
        candidates.append(id)

candidates = set(candidates)
print(query)
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
    cand_dist.append(distance[cand])
    weights.append(mp_prob(distance[cand]))
weights = 1.0 / np.array(weights)

dists = list(bins)
probs = []
for i in dists:
    probs.append( mp_prob(i))
matplotlib.rcParams.update({'font.size': 12})
gt, _ = np.histogram(distance, range=(lb,ub), bins=bins)
esti, _ = np.histogram(cand_dist, bins=bins, range=(lb, ub), weights=weights)
# print(np.sum(weights))
plt.legend(loc="upper right")
ax1 = plt.gca()
ax1.set_xlabel('distance')
ax1.set_ylabel('# of points')
ax1.plot(dists[:-1], gt, label="ground truth")
ax1.plot(dists[:-1], esti, label="mle")
plt.legend()
ax2 = ax1.twinx()
ax2.plot(dists, probs)
ax2.set_ylabel('recall')
plt.show()
# print(esti, len(candidates))
mre_na = np.abs(gt - esti) / gt
mre_na[np.isnan(mre_na)] = 0
mre_na[np.isinf(mre_na)] = 0

exp_path = pathlib.Path(sys.argv[1]).parent
mre_path = exp_path / "summary" / "mle-mre-q{}.txt".format(query)
np.savetxt(mre_path, mre_na)
