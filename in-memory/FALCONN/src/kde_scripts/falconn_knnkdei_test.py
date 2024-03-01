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

with open(os.path.join(exp_path,  "config/kde_tbl8.json")) as fin:
    conf = json.load(fin)

conf["query mode"] = "knn and kde infer"
del conf["hash table parameters"][0]
conf["seed"] = 0x84f2c02b
conf["knn filename"] = os.path.join(exp_path,  "summary/knn_test.ivecs")
conf["kde filename"] = os.path.join(exp_path,  "summary/kde_test.fvecs")
del conf["gamma"][0:9]
del conf["gamma"][1:]
conf["result filename"] = os.path.join(exp_path,  "knn_kde_time4.txt")

config_file = os.path.join(exp_path,  "config/knnkdei_v1.json")
with open(config_file, "w") as fout:
    json.dump(conf, fout, indent=4)

subprocess.run([os.path.join(bin_path, "falconn"), "-cf", config_file])
