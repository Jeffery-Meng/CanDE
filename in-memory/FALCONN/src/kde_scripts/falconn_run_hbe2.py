import subprocess
import pathlib, math
import os, sys
import json
import numpy as np
import fvecs

from all_one import all_one

# Run ground truth for KDE.

exp_path = sys.argv[1]
exp_path = str(pathlib.Path(exp_path).resolve())
dataset = pathlib.Path(exp_path).name

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

q_left = 800 if dataset in ["gistN", "gloveN"] else 900
config_file = os.path.join(exp_path, "config/kde_sample{}.json".format(q_left))
with open(config_file) as fin:
    config = json.load(fin)

hbe_config = {}

def write_hbe_config(fout, cfg):
    fout.write("gaussian {\n")
    for key, val in cfg.items():
        fout.write('\t{} = "{}";\n'.format(key, val))
    fout.write("}")


hbe_config["name"] = dataset
hbe_config["exact_path"] = ""
hbe_config["kernel"] = "gaussian"
hbe_config["fpath"] = str(pathlib.Path(exp_path).parent / "datasets/{}-train.csv".format(dataset))
hbe_config["qpath"] = str(pathlib.Path(exp_path).parent / "datasets/{}-test.csv".format(dataset))
# just to make sure HBS is run under default parameters
hbe_config["bw_const"] = "true"
hbe_config["tau"] = config["tau"]
hbe_config["d"] = config["dimension"]
hbe_config["n"] = config["training size"]
hbe_config["m"] = config["testing size"]
hbe_config["ignore_header"] = "true"
# start from 0, inclusive
hbe_config["start_col"] = 0
hbe_config["end_col"] = config["dimension"] - 1
# default setting
hbe_config["eps"] = 0.5
hbe_config["beta"] = 0.5

def midpoint(arr):
    return arr[len(arr)//2]

hbe_config["h"] = midpoint(config["gamma"])

hbe_config_path = os.path.join(exp_path, "config", "hbe.txt")
with open(hbe_config_path, "w") as fout:
    write_hbe_config(fout, hbe_config)

# hbe scope
config["dataset name"] = "gaussian"

# create kde folder
(pathlib.Path(exp_path) / "kde").mkdir(exist_ok=True)
config["result filename"] = os.path.join(exp_path, "kde/hbe{}_mres_v3.txt".format(q_left))
hbe_json_path = os.path.join(exp_path, "config", "hbe.json")
with open(hbe_json_path, "w") as fout:
    json.dump(config, fout, indent=4)

subprocess.run([os.path.join(bin_path, "hbe2"), hbe_config_path, hbe_json_path])