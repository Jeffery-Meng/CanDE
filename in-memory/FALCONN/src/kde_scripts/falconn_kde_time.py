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

config_file = os.path.join(exp_path, "config/kde_hist900.json")
with open(config_file) as fin:
    config = json.load(fin)

steps = 131072
table_widths = list(range(2, 16))

if os.path.exists("/media/gtnetuser/SSD_2TB_ALPHA"):
    dataset = pathlib.Path(exp_path).name
    config["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
    # config["candidate filename"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/candidates/dup_500_".format(dataset)
    for key, val in config.items():
        if isinstance(val, str):
            val = val.replace("/media/mydrive/KroneckerRotation", "/media/gtnetuser/SSD_2TB_ALPHA")
            val = val.replace("/media/mydrive/distribution/", "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/")
        config[key] = val


tw = 1
config["hash table width"] =tw * steps
config["result filename"] = os.path.join(exp_path, "summary",
        "timetest_{}_".format(tw * steps))
config["row id filename"] = os.path.join(exp_path, "summary",
        "histogram_timetest_")
config["histogram filename"] =  os.path.join(exp_path, "summary",
        "timetest_hist_bit.dvecs")
config["recall p filename"] =  os.path.join(exp_path, "summary",
            "timetest_recall_")
conf_path = os.path.join(exp_path, "config", "kde_timetest_{}.json".format(
    tw * steps))
with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_disthist_tbl2"), conf_path])
