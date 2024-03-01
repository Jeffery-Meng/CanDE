import subprocess
import pathlib
import os, sys
import json
from copy import deepcopy
import numpy as np

TargetRecall = 0.8
if "--targetrecall" in sys.argv:
    idx = sys.argv.index("--targetrecall")
    TargetRecall = float(sys.argv[idx+1])

exp_path = sys.argv[1]

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

config_file = os.path.join(exp_path, "ann_cd.json")
with open(config_file) as fin:
    config = json.load(fin)

config["eigenvalue filepath"] = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/mnist/gaussian_cdf.fvecs"
if not os.path.exists(config["eigenvalue filepath"]):
    from gaussian_cdf import *
    print_cdf(config["eigenvalue filepath"])
exp_path_p = pathlib.Path(exp_path)
m = config["hash table parameters"][0]["k"]
config["probing sequence file"] = str((exp_path_p / "probing_sequence.ivecs").resolve())
config["mp prob filename"] = str((exp_path_p / "mp_probs_{}.fvecs".format(m)).resolve())
config["hash function filename"] = str((exp_path_p / "hash_function.fvecs").resolve())
config["number of experiments"] = 1000
if "candidate filename" not in config:
    config["candidate filename"] = config["result filename"]

config["query mode"] = "knn duplicate candidates"
config_cd_file = os.path.join(exp_path, "ann_cd.json")
#with open(config_cd_file, "w") as fout:
 #   json.dump(config, fout, indent=4)
#candidate_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
#    "-cf", config_cd_file])

config["query mode"] = "precomputed sequence"
config_ps_file = os.path.join(exp_path, "ann_ps.json")
with open(config_ps_file, "w") as fout:
    json.dump(config, fout, indent=4)
#probings_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
#    "-cf", config_ps_file])

config["query mode"] = "hashed queries"
config_hq_file = os.path.join(exp_path, "ann_hq.json")
with open(config_hq_file, "w") as fout:
    json.dump(config, fout, indent=4)
#hashquery_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
 #   "-cf", config_hq_file])

config["query mode"] = "probing sequence"
config_ps2_file = os.path.join(exp_path, "ann_ps2b.json")
config["probing sequence file"] = str((exp_path_p / "probing_sequence2.ivecs").resolve())
with open(config_ps2_file, "w") as fout:
    json.dump(config, fout, indent=4)
#probings2_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
#    "-cf", config_ps2_file])

config["query mode"] = "hash function"
config_hf_file = os.path.join(exp_path, "ann_hf.json")
with open(config_hf_file, "w") as fout:
    json.dump(config, fout, indent=4)
#hf_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
 #   "-cf", config_hf_file])
#candidate_process.communicate()
#probings_process.communicate()
#probings2_process.communicate()
#hf_process.communicate()
#hashquery_process.communicate()

#recall_process = subprocess.Popen([os.path.join(bin_path, "falconn_recall_table"),
 #   config_cd_file])
#recall_process.communicate()
recall_summary_path = str((exp_path_p / "summary" / "recalls.txt").resolve())

with open(recall_summary_path) as fin:
    recalls = [float(x) for x in fin.readline().split()]
active_tables = np.searchsorted(recalls, TargetRecall) + 1
if active_tables > config["hash table parameters"][0]["l"]:
    active_tables = config["hash table parameters"][0]["l"]

config["hash table parameters"].append(deepcopy(config["hash table parameters"][0]))
config["hash table parameters"][1]["l"] = int(active_tables)
config["probing sequence file"] = str((exp_path_p / "probing_sequence2.ivecs").resolve())
config["result filename"] = str((exp_path_p / "recall_qh.fvecs").resolve())

#mp_process = subprocess.Popen([os.path.join(bin_path, "mp_probs"),
#    config_ps_file])
with open(config_ps2_file, "w") as fout:
    json.dump(config, fout, indent=4)
config["result filename"] = str((exp_path_p / "recall_qh2.fvecs").resolve())
config_ps3_file = os.path.join(exp_path, "ann_ps3b.json")
with open(config_ps3_file, "w") as fout:
    json.dump(config, fout, indent=4)
selectivity_process = subprocess.Popen([os.path.join(bin_path, "falconn_selectivity"),
    config_ps3_file])
#mp_process.communicate()
selectivity_process.communicate()

# qh_process = subprocess.Popen([os.path.join(bin_path, "recall_qh"),
#     config_ps2_file])
# qh2_process = subprocess.Popen([os.path.join(bin_path, "recall_qh2"),
#     config_ps3_file])
# qh_process.communicate()
# qh2_process.communicate()

