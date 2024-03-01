"""
    This script automatically fills the json config from the data and query sets.
"""

import sys
import json, os.path
from fvecs import *
import numpy as np
import shutil
from pathlib import Path

dataset_name = sys.argv[1]
exp_path = sys.argv[2]
data_file = sys.argv[3]
query_file = sys.argv[4]

exp_dataset_path = os.path.join(exp_path, dataset_name)
Path(exp_dataset_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(exp_dataset_path, "config")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(exp_dataset_path, "summary")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(exp_dataset_path, "candidates")).mkdir(parents=True, exist_ok=True)
config_file = os.path.join(exp_dataset_path, "ann.json")
config = {}

data = fvecs_read(data_file)
n, dim = data.shape
assert(n > dim)  # check data is not inverted
query = fvecs_read(query_file)
nq, dimq = query.shape
assert(dim == dimq)

# use only first 1000 queries, to avoid blow-up
if nq > 1000:
    nq = 1000
    query = query[:1000, :]
    query_file = "{}1K-test.fvecs".format(query_file.split("-")[0])
    to_fvecs(query_file, query)

config["data filename"] = data_file
config["query filename"] = query_file
config["dimension"] = dim
config["training size"] = n
config["testing size"] = nq
config["query mode"] = "knn recall"
config["hash table width"] = 20
config["load index"] = False
config["index path"] = "/dev/null"
config["number of partitions"] = 1
config["allow overwrite"] = True
config["compute ground truth"] = False
config["number of neighbors"] = 100
config["probes per table"] = 100
config["printing mode"] = {
        "id": True,
        "table": True,
        "bucket": True,
        "distance": False
    }
config["hash table parameters"] = [
        {
            "k": 8,
            "l": 20,
            "bucket width": 100000
        }
    ]
config["query hash filename"] = os.path.join(exp_dataset_path, "query_hash.fvecs")
config["candidate filename"] = os.path.join(exp_dataset_path, "candidates", "ex1")
config["kernel filename"] = os.path.join(exp_dataset_path, "weight.fvecs")
if ("--iskde" in sys.argv):
    from all_one import all_one
    # write all-one weight file
    all_one(n, config["kernel filename"])
config["summary path"] = os.path.join(exp_dataset_path, "summary") + "/"
config["ground truth file"] = os.path.join(exp_dataset_path, "gt.ivecs")
config["distance file"] = os.path.join(exp_dataset_path, "distances.fvecs")


def distances(data, query):
    dist = np.empty((nq, n))
    for qid in range(nq):
        for nid in range(n):
            dist[qid, nid] = np.linalg.norm(data[nid, :] - query[qid, :])
    return dist

def orders(dist):
    return np.argsort(dist, axis=1)

if not os.path.exists(config["distance file"]):
    dist = distances(data, query)
    to_fvecs(config["distance file"], dist)
else:
    dist = fvecs_read(config["distance file"])

if not os.path.exists(config["ground truth file"]):
    ords = orders(dist)
    to_ivecs(config["ground truth file"], ords)

with open(config_file, "w") as fout:
    json.dump(config, fout, indent=4)
