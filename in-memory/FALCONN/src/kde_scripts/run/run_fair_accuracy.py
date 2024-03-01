import os, subprocess, pathlib, numpy
import json, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["audio"]
#["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_fair"
processes = []

for dataset in datasets:
    cur_exp_path = os.path.join(exp_path, dataset)
    with open(os.path.join(cur_exp_path, "ann_cd.json")) as fin:
        conf = json.load(fin)
    with open(os.path.join(cur_exp_path, "ann_ps3.json")) as fin:
        conf2 = json.load(fin)
    conf["hash table parameters"].append(
        conf2["hash table parameters"][1])
    conf["max attempts"] = 10000
    fair_folder = os.path.join(cur_exp_path, "fair_nn")
    pathlib.Path(fair_folder).mkdir(parents=True, exist_ok=True)
    conf["result filename"] = os.path.join(fair_folder, "fair.txt")
    distance = fvecs.fvecs_read(conf["distance file"])
    quantiles = [10, 30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    conf["nn radius"] = [float(numpy.quantile(distance, x/conf["training size"])) for x in quantiles]
    conf["number of experiments"] = 100000

    fair_conf_path = os.path.join(cur_exp_path, "fair.json")
    with open(fair_conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)

    processes.append(subprocess.Popen([script_path, fair_conf_path, "10"]))
    
for process in processes:
    process.communicate()

