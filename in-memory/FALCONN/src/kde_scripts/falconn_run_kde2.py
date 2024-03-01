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

config_file = os.path.join(exp_path, "config/kde_tbl3.json")
with open(config_file) as fin:
    config = json.load(fin)

nb_range = [1, 4, 20]  # log scale from 1 to 10000, 20 splits


config["result filename"] = os.path.join(exp_path, "summary",
        "histogram_time_")
config["row id filename"] = os.path.join(exp_path, "summary",
        "histogram_mres_")
config["histogram filename"] =  os.path.join(exp_path, "summary",
        "kde_histogram_sketch_")
config["recall p filename"] =  os.path.join(exp_path, "summary",
            "sketch_recall2_")
config["hash table width"] = 2097152
conf_path = os.path.join(exp_path, "config", "kde_hist.json")
with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_disthist_sketch"), conf_path])


config["result filename"] = os.path.join(exp_path,
        "kde_sketch_mres_")
config["row id filename"] = os.path.join(exp_path, "selected_queries.ivecs")
conf_path = os.path.join(exp_path, "config", "kde_tbl4.json")

with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_kde_v3"), conf_path])
