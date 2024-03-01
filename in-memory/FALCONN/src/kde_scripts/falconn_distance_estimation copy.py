import subprocess
import pathlib
import os, sys
import json

exp_path = sys.argv[1]
idx = sys.argv.index("--targetrecall")
targ_recall = float(sys.argv[idx+1])

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

config_file = os.path.join(exp_path, "ann.json")
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
config["number of experiments"] = 1000
if "candidate filename" not in config:
    config["candidate filename"] = config["result filename"]

config["query mode"] = "knn candidates"
config_cd_file = os.path.join(exp_path, "ann_cd.json")
with open(config_cd_file, "w") as fout:
    json.dump(config, fout, indent=4)
candidate_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
    "-cf", config_cd_file])

config["query mode"] = "precomputed sequence"
config_ps_file = os.path.join(exp_path, "ann_ps.json")
with open(config_ps_file, "w") as fout:
    json.dump(config, fout, indent=4)
probings_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
    "-cf", config_ps_file])

candidate_process.communicate()
probings_process.communicate()
recall_process = subprocess.Popen([os.path.join(bin_path, "falconn_recall_table"),
    config_cd_file])
if not os.path.exists(config["mp prob filename"]):
    mp_process = subprocess.Popen([os.path.join(bin_path, "mp_probs"),
        config_ps_file])
    mp_process.communicate()
recall_process.communicate()

# find number of hash tables for achieving target recall
with open(os.path.join(config["summary path"], "recalls.txt")) as fin:
    recalls = [float(x) for x in fin.readline().split()]
    cand_ratios = [float(x) for x in fin.readline().split()]
l = next(idx for idx, val in enumerate(recalls) if val > targ_recall) + 1
print(l, recalls[l-1], cand_ratios[l-1])
config["hash table parameters"].append(config["hash table parameters"][-1].copy())
config["hash table parameters"][-1]["l"] = l
config_de_file = os.path.join(exp_path, "ann_de.json")
with open(config_de_file, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run(["python3", os.path.join(this_path, "distance_hist_queries.py"),
    config_de_file, "-l", str(l)])
