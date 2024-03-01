import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fvecs import *
import sys

dataset = sys.argv[1]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/" + dataset
conf_path = exp_path + "/ann_ps3.json"
Bay_weight = 250
qid = int(sys.argv[2])

with open(conf_path) as fin:
    conf = json.load(fin)
l = conf["hash table parameters"][1]['l']
x_bins = np.arange(*conf["histogram bins"])[:-1]
mre = np.zeros(len(x_bins))
np.seterr(all="ignore")
distance = fvecs_read(exp_path + "/distances.fvecs")
#recall_qh_prior = fvecs_read(conf["result filename"])
# recall_qh_prior = recall_qh_prior[qid, :]
distance = distance[qid, :]

from read_candidate import load_candidate

def recall_prob(table_probs):
    prod = 1.0
    for p in table_probs:
        prod *= 1 - p
    return 1.0 - prod

def estimate(query, bins):
    candiate_tables = load_candidate(query, conf_path)[:l]
    flat = []
    for ct in candiate_tables:
        flat.extend(ct)
    counters = np.zeros(1000000, dtype=np.int32)
    first_occur = np.full(1000000, -1, dtype=np.int32)
    for table, ct in enumerate(candiate_tables):
        for cand in ct:
            counters[cand] += 1
            if first_occur[cand] == -1:
                first_occur[cand] = table
    candidates = set(flat)

    dist_f = pd.DataFrame({"id": list(range(1000000)), "dist": distance,
            "count": counters, "table": first_occur})
    distance_group, bins = pd.cut(dist_f["dist"], bins=np.arange(*bins), retbins = True)
    dist_f["dist group"] = distance_group
    cand_f = dist_f.loc[candidates]
    gt_count = dist_f.groupby("dist group").count()
    gt_count = np.array(gt_count["id"])
    cand_f = cand_f.groupby("dist group")
    cand_count = cand_f.count()
    cand_count = np.array(cand_count["id"])
    recall_qh = np.zeros(len(cand_count), dtype=np.float64)
    effective_sizes = np.zeros(len(cand_count), dtype=np.int32)
    
    for bin_id, cand_idx in enumerate(cand_f.groups):
        try:
            cand_in_bin = cand_f.get_group(cand_idx)
        except KeyError:
            continue
        recalls = np.zeros(l)
        for table in range(l):
            effective_rows = (cand_in_bin["count"] > 1) | (cand_in_bin["table"] != table)
            cand_effective = set(cand_in_bin[effective_rows]["id"])
            denom = len(cand_effective)
            if denom == 0:
                continue
            numer = len(cand_effective & set(candiate_tables[table]))
            recalls[table] =  numer / denom

            effective_sizes[bin_id] += denom
        recall_qh[bin_id] = recall_prob(recalls)


    bucket_width = conf["hash table parameters"][0]["bucket width"]

    mp_tuple, mp_recalls = tuple(fvecs_read_non_uniform(conf["mp prob filename"]))
    _, _, mpstep = mp_tuple
    mp_recalls = 1.0 - (1.0 - mp_recalls) ** l
    mp_size = len(mp_recalls)
    weights = np.zeros(len(cand_count), dtype=np.float64)
    

    def mp_prob(distance):
        idx = int(distance / bucket_width / mpstep)
        if idx < 0:
            ids = 0
        if idx >= mp_size:
            idx = mp_size - 1
        return mp_recalls[idx]

    for bin_id, cand_idx in enumerate(cand_f.groups):
        bin_dist = (bins[bin_id] + bins[bin_id + 1]) / 2
        weights[bin_id] = mp_prob(bin_dist)

    return gt_count, cand_count / recall_qh, cand_count / weights, effective_sizes
    


import matplotlib
matplotlib.rcParams.update({'font.size': 12})

print(qid)
gt, esti, mle, effective_sizes = estimate(qid, conf["histogram bins"])
esti[np.isnan(esti)] = 0
esti[np.isinf(esti)] = 0
plt.clf()
ax = plt.gca()

ax.plot(x_bins, mle, label="basic")
ax.plot(x_bins, esti, label="alternative")
ax.plot(x_bins, gt, label="ground truth")
ax.set_xlabel("distance")
ax.set_ylabel("Number of points")
ax.legend(loc="upper right")
# plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plots/{}_bayesian_{}.pdf".format(
#        dataset, qid))
mre_na = np.abs(gt - esti) / gt
mre_na[np.isnan(mre_na)] = 0
mre_na[np.isinf(mre_na)] = 0
mre_path = exp_path + "/summary/combined-mre-q{}.txt".format(qid)
eff_path = exp_path + "/summary/effsize-q{}.txt".format(qid)
np.savetxt(mre_path, mre_na)
np.savetxt(eff_path, effective_sizes)

