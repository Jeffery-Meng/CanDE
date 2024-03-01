import subprocess
import pathlib
import os, sys
import json
import numpy as np
import fvecs

from all_one import all_one

# Run ground truth for KDE.

exp_path = sys.argv[1]
exp_path = str(pathlib.Path(exp_path).resolve())

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

config_file = os.path.join(exp_path,  "config/kde_timetest_131072.json") 
#"config/kde_tbl_100.json")
with open(config_file) as fin:
    config = json.load(fin)

"""
if os.path.exists("/media/gtnetuser/SSD_2TB_ALPHA"):
    dataset = pathlib.Path(exp_path).name
    config["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
    # config["candidate filename"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/candidates/dup_500_".format(dataset)
    for key, val in config.items():
        if isinstance(val, str):
            val = val.replace("/media/mydrive/KroneckerRotation", "/media/gtnetuser/SSD_2TB_ALPHA")
            val = val.replace("/media/mydrive/distribution/", "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/")
        config[key] = val
"""

config_kde_file = os.path.join(exp_path, "kde.json")
config["ground truth file"] = os.path.join(exp_path, "kde_gt900.dvecs")
config["result filename"] = os.path.join(exp_path, "kde_mle900.txt")
config["row id filename"] = os.path.join(exp_path, "selected_queries900.ivecs")

if not config.get("kernel filename", "") or not os.path.exists(config["kernel filename"]):
    config["kernel filename"] = os.path.join(exp_path, "weights.fvecs")
    all_one(config["training size"], config["kernel filename"])

with open(config_kde_file, "w") as fout:
    json.dump(config, fout, indent=4)

#use given gamma values
proc = subprocess.Popen([os.path.join(bin_path, "kde_gt"),
    config_kde_file, "true"], stdout=subprocess.PIPE)
gammas = [float(x) for x in proc.stdout.readline().split()]
proc.communicate()

config["row id filename"] = os.path.join(exp_path, "selected_queries900.ivecs")
config["result filename"] = os.path.join(exp_path,
        "kde_hash_")
config["recall p filename"]  = os.path.join(exp_path,
        "summary/timetest_recall_")
conf_path = os.path.join(exp_path, "config", "kde_tbl8.json")

with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_kde_v3"), conf_path])
