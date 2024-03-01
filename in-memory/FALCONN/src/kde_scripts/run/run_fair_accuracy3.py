import os, subprocess, pathlib, numpy, errno
import json, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs
import time

def is_running(pid):        
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
    return True

# Warning: This PID is valid only for this time
# Wait the previous experiment to end, so that we do not blow up memory.
while is_running(35034):
    time.sleep(30)

datasets = ["mnist"]  # deep and sift
num_exps = 100000
num_tables=150
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_fair"
sample_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/sample"
processes = []


for dataset in datasets:
    cur_exp_path = os.path.join(exp_path, dataset)
    with open(os.path.join(cur_exp_path, "ann_cd.json")) as fin:
        conf = json.load(fin)
        
    for key, val in conf.items():
        if isinstance(val, str):
            # print(val)
            val = val.replace("/media/gtnetuser/Elements/QDDE", "/media/gtnetuser/SSD_2TB_BEST/QDDE")
            conf[key] = val
        
    with open(os.path.join(cur_exp_path, "ann_ps3b.json")) as fin:
        conf2 = json.load(fin)
    conf["hash table parameters"].append(
        conf2["hash table parameters"][1])
    conf["max attempts"] = 100000

    with open(os.path.join(sample_path, dataset, "fairnn_sample.json")) as fin:
        conf3 = json.load(fin) 

    
    conf["nn radius"] = conf3["nn radius"]

    conf["number of experiments"] = num_exps
    
    conf["hash table parameters"][0]["l"] = 150
    
    for i in range(5):
        conf["row id filename"] = os.path.join(sample_path, dataset, "fair_sample/sample_id_" + str(i)+".fvecs")
        conf["result filename"] = os.path.join(cur_exp_path, "fair_nn/CanDE_new_more_"+str(i)+".csv")
        fair_conf_path = os.path.join(cur_exp_path, "config", "fair_falconn_part_more_" + str(i)+".json")
        with open(fair_conf_path, "w") as fout:
            json.dump(conf, fout, indent=4)

        processes.append(subprocess.Popen([script_path, fair_conf_path, "10"]))

for process in processes:
    process.communicate()

