import subprocess
import pathlib
import os, sys
import json

exp_path = sys.argv[1]

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

config_file = os.path.join(exp_path, "ann.json")
with open(config_file) as fin:
    config = json.load(fin)

config["ground truth file"] = os.path.join(exp_path, "svmgt.fvecs")
config["eigenvalue filepath"] = os.path.join(exp_path, "gaussian_cdf.fvecs")
if not os.path.exists(config["eigenvalue filepath"]):
    from gaussian_cdf import *
    print_cdf(config["eigenvalue filepath"])
config["row id filename"] = os.path.join(exp_path, "relative_error.txt")
config["probing sequence file"] = os.path.join(exp_path, "probing_sequence.ivecs")
config["recall p filename"] =  os.path.join(exp_path, "recall_p.fvecs") 

config["query mode"] = "knn candidates"
config_cd_file = os.path.join(exp_path, "ann_cd.json")
with open(config_cd_file, "w") as fout:
    json.dump(config, fout, indent=4)
candidate_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
    "-cf", config_cd_file])

config["query mode"] = "hashed queries"
config_hq_file = os.path.join(exp_path, "ann_hq.json")
with open(config_hq_file, "w") as fout:
    json.dump(config, fout, indent=4)
queryh_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
    "-cf", config_hq_file])

config["query mode"] = "probing sequence"
config_ps_file = os.path.join(exp_path, "ann_ps.json")
with open(config_ps_file, "w") as fout:
    json.dump(config, fout, indent=4)
probings_process = subprocess.Popen([os.path.join(bin_path, "falconn"),
    "-cf", config_ps_file])

candidate_process.communicate()
queryh_process.communicate()
probings_process.communicate()

subprocess.run([os.path.join(bin_path, "candidate_probs"),
    config_cd_file])