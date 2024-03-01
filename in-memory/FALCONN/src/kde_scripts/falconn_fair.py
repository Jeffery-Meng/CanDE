import subprocess
import pathlib
import os, sys
import json
from copy import deepcopy
import numpy as np

exp_path = sys.argv[1]
num_tables = sys.argv[2]

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")
exp_split = exp_path.split("/")
dataset = exp_split[-1] if exp_split[-1] else exp_split[-2]
external_path = pathlib.Path("/media/gtnetuser/Elements/QDDE") / dataset

config_file = os.path.join(exp_path, "ann.json")
with open(config_file) as fin:
    config = json.load(fin)

config["eigenvalue filepath"] = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/mnist/gaussian_cdf.fvecs"
if not os.path.exists(config["eigenvalue filepath"]):
    from gaussian_cdf import *
    print_cdf(config["eigenvalue filepath"])
exp_path_p = pathlib.Path(exp_path)
m = config["hash table parameters"][0]["k"]
config["hash table parameters"][0]["l"] = int(num_tables)
config["probing sequence file"] = str((exp_path_p / "probing_sequence.ivecs").resolve())
config["mp prob filename"] = str((exp_path_p / "mp_probs_{}.fvecs".format(m)).resolve())
config["hash function filename"] = str((exp_path_p / "hash_function.fvecs").resolve())
config["number of experiments"] = 1000
config["candidate filename"] = str((external_path / "candidates" / "dup_{}_".format(num_tables)).resolve())

config["query mode"] = "knn duplicate candidates"
config_cd_file = os.path.join(exp_path, "ann_cd.json")

external_path.mkdir(parents=True, exist_ok=True)
(external_path / "index").mkdir(parents=True, exist_ok=True)
(external_path / "candidates").mkdir(parents=True, exist_ok=True)
config["index path"] = str(external_path / "index") + "/"
config["index filename"] = "knn_{}.txt".format(num_tables)
with open(config_cd_file, "w") as fout:
    json.dump(config, fout, indent=4)
candidate_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
    "-cf", config_cd_file])
candidate_process.communicate()

config["index filename"] = ""
if not os.path.exists(config["mp prob filename"]):
    config["query mode"] = "precomputed sequence"
    config_ps_file = os.path.join(exp_path, "ann_ps.json")
    with open(config_ps_file, "w") as fout:
        json.dump(config, fout, indent=4)
    probings_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
        "-cf", config_ps_file])
    mp_process = subprocess.Popen([os.path.join(bin_path, "mp_probs"),
        config_ps_file])
    mp_process.communicate()
