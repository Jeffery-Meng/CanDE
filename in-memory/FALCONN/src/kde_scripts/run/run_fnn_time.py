import os, subprocess, pathlib, numpy
import json, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["glove","sift"] # "enron","trevi", "audio", "mnist"
exp_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/FALCONN/build/falconn_fair_time"
fairnn_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/FALCONN/build/fair_nn_time"
sample_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/EXPERIMENTS/sample"
processes = []

ls_all = [[25,42,100],[30,46,100]]

for j in range(len(datasets)):
    dataset = datasets[j]
    ls = ls_all[j]
    for i in range(len(ls)):
        l = ls[i]
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
        conf["hash table parameters"][1] = conf2["hash table parameters"][1]
        if l >  0:
            conf["hash table parameters"][1]["l"] = l

        conf["max attempts"] = 100000
        fair_folder = os.path.join(cur_exp_path, "fair_nn")
        pathlib.Path(fair_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(sample_path, dataset, "fairnn_sample.json")) as fin:
            conf3 = json.load(fin) 

        conf["nn radius"] = conf3["nn radius"][i]
        conf["row id filename"] = os.path.join(sample_path, dataset, "fair_sample/sample_id.fvecs")
        conf["result filename"] = os.path.join(fair_folder, "fair_time_" + str(conf["hash table parameters"][1]["l"])+".csv")
        conf["number of experiments"] = 100

        fair_conf_path = os.path.join(cur_exp_path,  "fairnn_time_{}.json".format(conf["hash table parameters"][1]["l"]))
        with open(fair_conf_path, "w") as fout:
            json.dump(conf, fout, indent=4)

        subprocess.run([fairnn_path, fair_conf_path, "1"])
    
