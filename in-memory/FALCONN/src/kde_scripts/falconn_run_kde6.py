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

config_file = os.path.join(exp_path, "ann_ps3b.json")
with open(config_file) as fin:
    config = json.load(fin)

from scott_rot import scott_rot

scott = scott_rot(config["data filename"])[0]
print("Scott rule of thumb: ", scott)
gammas = [0.5, 10] # doubles every 10 values
gammas = [scott * x for x in gammas]
gammas.extend([20, 1.0, 0.82, 0.8])
# 20 gamma values, the last is 3 times the first, select 950 to 900 queries
config["gamma"] =  gammas
config["tau"] = 1e-6
config["dataset"] = pathlib.Path(sys.argv[1]).name
config_kde_file = os.path.join(exp_path, "kde.json")
config["ground truth file"] = os.path.join(exp_path, "kde_gt800.dvecs")
config["result filename"] = os.path.join(exp_path, "kde_mle800.txt")
config["row id filename"] = os.path.join(exp_path, "selected_queries800.ivecs")

if not config.get("kernel filename", "") or not os.path.exists(config["kernel filename"]):
    config["kernel filename"] = os.path.join(exp_path, "weights.fvecs")
    all_one(config["training size"], config["kernel filename"])

with open(config_kde_file, "w") as fout:
    json.dump(config, fout, indent=4)

proc = subprocess.Popen([os.path.join(bin_path, "kde_gt"),
    config_kde_file], stdout=subprocess.PIPE)
gammas = [float(x) for x in proc.stdout.readline().split()]
proc.communicate()

with open(os.path.join(exp_path, "ann_ps3b.json")) as fin:
    conf2 = json.load(fin)
config["hash table parameters"] = conf2["hash table parameters"]
config["result filename"] = os.path.join(exp_path, "kde_mle800b.txt")
config["gamma"] = gammas
conf_path = os.path.join(exp_path, "kde_mle800.json")
with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_kde"), conf_path])

with open(os.path.join(exp_path, "config", "kde_hist.json")) as fin:
    conf_hist = json.load(fin)

config["histogram bins"] = conf_hist["histogram bins"]
config["result filename"] = os.path.join(exp_path, "summary",
        "histogram_time800_")
config["row id filename"] = os.path.join(exp_path, "summary",
        "histogram_mres800_")
config["histogram filename"] =  os.path.join(exp_path, "summary",
        "kde_histogram800_bit.dvecs")
config["recall p filename"] =  os.path.join(exp_path, "summary",
            "bit_recall800_")
config["hash table width"] = 1048576
conf_path = os.path.join(exp_path, "config", "kde_hist800.json")
with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_disthist_sketch"), conf_path])

config["row id filename"] = os.path.join(exp_path, "selected_queries800.ivecs")
config["result filename"] = os.path.join(exp_path,
        "kde_sketch_mres800_")
conf_path = os.path.join(exp_path, "config", "kde_tbl5.json")

with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_kde_v3"), conf_path])

config["result filename"] = os.path.join(exp_path, "summary",
        "histogram_time800tbl_")
config["row id filename"] = os.path.join(exp_path, "summary",
        "histogram_mres800tbl_")
config["histogram filename"] =  os.path.join(exp_path, "summary",
        "kde_histogram800_tbl.dvecs")
config["recall p filename"] =  os.path.join(exp_path, "summary",
            "tbl_recall800_")
config["hash table width"] = 1048576
conf_path = os.path.join(exp_path, "config", "kde_hist800tbl.json")
with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_disthist_tbl2"), conf_path])

config["row id filename"] = os.path.join(exp_path, "selected_queries800.ivecs")
config["result filename"] = os.path.join(exp_path,
        "kde_tbl_mres800_")
conf_path = os.path.join(exp_path, "config", "kde_tbl7.json")

with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "falconn_kde_v3"), conf_path])

with open(os.path.join(exp_path, "../selectivity.txt")) as fin:
    for line in fin.readlines():
        dataset_name, l, sampling_ratio = tuple(line.split())
        if dataset_name != config["dataset"]:
            continue
        sampling_ratio = float(sampling_ratio)
        break
config["ratio of prefilter"] = sampling_ratio
conf_path = os.path.join(exp_path, "config", "kde_sample800.json")

config["result filename"] = os.path.join(exp_path, "kde_sample_mres800.txt")
with open(conf_path, "w") as fout:
    json.dump(config, fout, indent=4)
subprocess.run([os.path.join(bin_path, "kde_sample"), conf_path])

hbe_config = {}

def write_hbe_config(fout, cfg):
    fout.write("gaussian {\n")
    for key, val in cfg.items():
        fout.write('\t{} = "{}";\n'.format(key, val))
    fout.write("}")

dataset = pathlib.Path(exp_path).name
hbe_config["name"] = dataset
hbe_config["exact_path"] = ""
hbe_config["kernel"] = "gaussian"
hbe_config["fpath"] = str(pathlib.Path(exp_path).parent / "datasets/{}-train.csv".format(dataset))
hbe_config["qpath"] = str(pathlib.Path(exp_path).parent / "datasets/{}-test.csv".format(dataset))
# just to make sure HBS is run under default parameters
hbe_config["h"] = 1.0
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

hbe_config_path = os.path.join(exp_path, "config", "hbe.txt")
with open(hbe_config_path, "w") as fout:
    write_hbe_config(fout, hbe_config)

# hbe scope
config["dataset name"] = "gaussian"

# create kde folder
(pathlib.Path(exp_path) / "kde").mkdir(exist_ok=True)
config["result filename"] = os.path.join(exp_path, "kde/hbe800_mres.txt")
hbe_json_path = os.path.join(exp_path, "config", "hbe.json")
with open(hbe_json_path, "w") as fout:
    json.dump(config, fout, indent=4)

subprocess.run([os.path.join(bin_path, "hbe"), hbe_config_path, hbe_json_path])