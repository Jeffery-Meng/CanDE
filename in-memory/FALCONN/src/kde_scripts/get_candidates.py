import subprocess
import pathlib
import os, sys
import json
import numpy as np
import fvecs

from all_one import all_one

# Run ground truth for KDE.

datasets = ["audio"]
tables = [1000]
#["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path_org = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/EXPERIMENTS/"
program_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/FALCONN/build/falconn_kde"
modes = ["Print candidates number"] # "knn and kde associative",
result_ends = ["none"] # "infer","associative",


for j in range(len(datasets)):
    for i in range(len(modes)):
        dataset = datasets[j]
        table = tables[j]
        exp_path = os.path.join(exp_path_org,dataset)
        with open(os.path.join(exp_path, "config/kde_hist900.json")) as fin:
            conf = json.load(fin)
            
            for key, val in conf.items():
                if isinstance(val, str):
                        # print(val)
                    val = val.replace("/media/gtnetuser/Elements/QDDE", "/media/gtnetuser/SSD_2TB_BEST/QDDE")
                    val = val.replace("/media/mydrive/KroneckerRotation/data/", "/media/gtnetuser/SSD_2TB_ALPHA/data/")
                    val = val.replace("/media/mydrive/distribution/","/media/gtnetuser/SSD_2TB_ALICE/distribution_new/")
                    conf[key] = val
                
        
        conf["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
        conf["hash table parameters"][1]["l"] = table  
        conf["query mode"] = modes[i]
        conf["hash table width"] = 20
        # conf["ground truth file"] = os.path.join(exp_path, "kde_gt900.dvecs")
        
        del conf["hash table parameters"][0]
        conf["seed"] = 0x84f2c02b
        conf["knn filename"] = os.path.join(exp_path,  "summary/knn_test" + result_ends[i]+ ".ivecs")
        conf["kde filename"] = os.path.join(exp_path,  "summary/kde_test" + result_ends[i]+ ".fvecs")
        conf["cande table width"] = 17
        del conf["gamma"][0:9]
        del conf["gamma"][1:]
        conf["result filename"] = os.path.join(exp_path,  "knn_kde_" + result_ends[i]+  ".txt")


        config_file = os.path.join(exp_path,  "config/knnkdei_final_"+result_ends[i]+ ".json")
        with open(config_file, "w") as fout:
            json.dump(conf, fout, indent=4)

        subprocess.run([program_path, "-cf", config_file])