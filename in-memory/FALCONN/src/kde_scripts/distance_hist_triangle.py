import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fvecs import *
import sys

dataset = sys.argv[1]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/" + dataset
conf_path = exp_path + "/ann_ps2.json"
Bay_weight = 1000
qid = int(sys.argv[2])

with open(conf_path) as fin:
    conf = json.load(fin)
l = conf["hash table parameters"][1]['l']
x_bins = np.arange(*conf["histogram bins"])[:-1]
mre = np.zeros(len(x_bins))
np.seterr(all="ignore")
distance = fvecs_read(exp_path + "/distances.fvecs")
recall_qh = fvecs_read(conf["result filename"])
recall_qh = recall_qh[qid, :]
distance = distance[qid, :]

from read_candidate import load_candidate


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
    return gt_count, cand_count / recall_qh[:-1]
    


print(qid)
gt, esti = estimate(qid, conf["histogram bins"])
esti[np.isnan(esti)] = 0
esti[np.isinf(esti)] = 0
# plt.clf()
# ax = plt.gca()
# ax.plot(x_bins, gt, label="ground truth")
# ax.plot(x_bins, esti, label="estimated")
# ax.set_xlabel("distance")
# ax.set_ylabel("Number of points")
# ax.legend(loc="upper right")
# plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plots/{}_bayesian_{}.pdf".format(
#         dataset, qid))
mre_na = np.abs(gt - esti) / gt
mre_na[np.isnan(mre_na)] = 0
mre_na[np.isinf(mre_na)] = 0
mre_path = exp_path + "/summary/triangle-mre-q{}.txt".format(qid)
np.savetxt(mre_path, mre_na)


