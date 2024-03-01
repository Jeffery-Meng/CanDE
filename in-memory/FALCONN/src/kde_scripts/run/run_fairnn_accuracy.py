import os, subprocess, pathlib, numpy
import json, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["audio"]
radius_list = [37000, 57000, 87000]
l_tables = [20, 40, 100, 200, 500]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/fair_nn"
processes = []

for dataset in datasets:
    for radius in radius_list:
        for l in l_tables:
            cur_exp_path = os.path.join(exp_path, dataset)
            with open(os.path.join(cur_exp_path, "fair.json")) as fin:
                conf = json.load(fin)
            with open(os.path.join(cur_exp_path, "fairnn_{}.json".format(radius))) as fin:
                conf2 = json.load(fin)
            conf["hash table parameters"][0] = conf2["hash table parameters"][0]
            conf["hash table parameters"][1] = conf2["hash table parameters"][0].copy()
            conf["hash table parameters"][1]["l"] = l
            conf["candidate filename"] = conf2["candidate filename"]
            conf["max attempts"] = 1000000
            fair_folder = os.path.join(cur_exp_path, "fair_nn")
            pathlib.Path(fair_folder).mkdir(parents=True, exist_ok=True)
            conf["result filename"] = os.path.join(fair_folder, "fairnn_{}_{}.csv".format(radius, l))
            # nn radius and number of experiments keep the same as in fair.json

            fair_conf_path = os.path.join(cur_exp_path, "config", "fairnn_{}_{}.json".format(radius, l))
            with open(fair_conf_path, "w") as fout:
                json.dump(conf, fout, indent=4)

            processes.append(subprocess.Popen([script_path, fair_conf_path, "10"]))
    
for process in processes:
    process.communicate()

