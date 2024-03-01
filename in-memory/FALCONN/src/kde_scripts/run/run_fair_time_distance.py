import os, subprocess, pathlib, numpy
import json, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["sift"] # "enron", "gist"
num_tables = [100]
starts = 0
ends = 4
exp_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/FALCONN/build/falconn_fair_time_distance"
fairnn_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/FALCONN/build/fair_nn_time"
sample_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/EXPERIMENTS/sample"
processes = []

for i in range(len(datasets)):
    dataset = datasets[i]
    num_table = num_tables[i]
    cur_exp_path = os.path.join(exp_path, dataset)
    with open(os.path.join(cur_exp_path, "fair.json")) as fin:
        conf = json.load(fin)
    
    conf["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
    conf["candidate filename"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/candidates/dup_500_".format(dataset)
    for key, val in conf.items():
        if isinstance(val, str):
            val = val.replace("/media/mydrive/KroneckerRotation", "/media/gtnetuser/SSD_2TB_ALPHA")
            val = val.replace("/media/mydrive/distribution/", "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/")
        conf[key] = val

    with open(os.path.join(cur_exp_path, "ann_ps3b.json")) as fin:
        conf2 = json.load(fin)

    conf["hash table parameters"][0]["l"] = num_table
    conf["hash table parameters"][1] = conf2["hash table parameters"][1]
    conf["max attempts"] = 100000
    fair_folder = os.path.join(cur_exp_path, "fair_nn")
    pathlib.Path(fair_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(sample_path, dataset, "fairnn_sample.json")) as fin:
        conf3 = json.load(fin) 

    conf["nn radius"] = conf3["nn radius"][starts:ends]
    conf["row id filename"] = os.path.join(sample_path, dataset, "fair_sample/sample_id.fvecs")
    conf["result filename"] = os.path.join(fair_folder, "falconn_fnn_time_distance.csv")
    conf["number of experiments"] = 100

    fair_conf_path = os.path.join(cur_exp_path, "falconn_fnn_time.json")
    with open(fair_conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)

    subprocess.run([script_path, fair_conf_path, "1"])

