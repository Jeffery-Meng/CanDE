import os, subprocess, pathlib, numpy
import json, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["glove","gist"]  # deep and sift, gist
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/fair_nn_mp" # fair_nn
sample_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/sample"
processes = []

ls = [-1,50,100]
for l in ls:
    for dataset in datasets:
        cur_exp_path = os.path.join(exp_path, dataset)
        with open(os.path.join(cur_exp_path, "fair.json")) as fin:
            conf = json.load(fin)
        
        with open(os.path.join(cur_exp_path, "ann_ps3b.json")) as fin:
            conf2 = json.load(fin)

        conf["hash table parameters"][1] =  conf2["hash table parameters"][1]
        if l > 0:
            conf["hash table parameters"][1]["l"] = l

        for key, val in conf.items():
            if isinstance(val, str):
                # print(val)
                val = val.replace("/media/gtnetuser/Elements/QDDE", "/media/gtnetuser/SSD_2TB_BEST/QDDE")

            
                conf[key] = val
                
        conf["max attempts"] = 100000
        fair_folder = os.path.join(cur_exp_path, "fair_nn")
        pathlib.Path(fair_folder).mkdir(parents=True, exist_ok=True)
        conf["result filename"] = os.path.join(fair_folder, "fairnn_{}.csv".format(l))
  
        with open(os.path.join(sample_path, dataset, "fairnn_sample.json")) as fin:
            conf3 = json.load(fin) 
        conf["nn radius"] = conf3["nn radius"]
        conf["row id filename"] = os.path.join(sample_path, dataset, "fair_sample/sample_id.fvecs")
        conf["number of experiments"] = 1000000


        fair_conf_path = os.path.join(cur_exp_path, "config", "fairnn_{}.json".format(l))
        with open(fair_conf_path, "w") as fout:
            json.dump(conf, fout, indent=4)

        processes.append(subprocess.Popen([script_path, fair_conf_path, "10"]))
    
        for process in processes:
            process.communicate()

