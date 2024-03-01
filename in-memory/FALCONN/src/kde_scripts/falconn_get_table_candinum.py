import subprocess
import pathlib
import os, sys
import json
import numpy as np
import fvecs

from all_one import all_one

# Run ground truth for KDE.

datasets = ["enron","trevi","mnist"]
#["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path_org = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
program_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn"
modes = ["Print candidates number"] 
result_ends = ["candidatesnum"]


for j in range(len(datasets)):
    for i in range(len(modes)):
        dataset = datasets[j]
        exp_path = os.path.join(exp_path_org,dataset)
        with open(os.path.join(exp_path, "ann_ps3b.json")) as fin:
            conf = json.load(fin)
            
            # for key, val in conf.items():
            #     if isinstance(val, str):
            #             # print(val)
            #         val = val.replace("/media/gtnetuser/16TB/QDDE/", "/media/gtnetuser/SSD_2TB_BEST/QDDE")
            #         val = val.replace("/media/mydrive/KroneckerRotation/data/", "/media/gtnetuser/SSD_2TB_ALPHA/data/")
            #         val = val.replace("/media/gtnetuser/Elements/data/", "/media/gtnetuser/SSD_2TB_ALPHA/data/")
            #         val = val.replace("/media/gtnetuser/Elements/distribution/","/media/gtnetuser/SSD_2TB_ALICE/distribution_new/")
            #         conf[key] = val
        
                   
        conf["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
        conf["query mode"] = modes[i]
        conf["hash table width"] = 20
        # conf["ground truth file"] = os.path.join(exp_path, "kde_gt900.dvecs")
        
        del conf["hash table parameters"][0]
        conf["testing size"] = 1
        conf["hash table parameters"][0]["l"] = 10000
        conf["seed"] = 0x84f2c02b
        conf["knn filename"] = os.path.join(exp_path,  "summary/knn_test_" + result_ends[i]+ ".ivecs")
        conf["kde filename"] = os.path.join(exp_path,  "summary/kde_test_" + result_ends[i]+ ".fvecs")
        conf["cande table width"] = 18
        # if len(conf["histogram bins"]) == 3:
        #     conf["histogram bins"].append(50)
        conf["result filename"] = os.path.join(exp_path,  "knn_qdde_" + result_ends[i]+  ".txt")
        conf["candidate filename"] = os.path.join(exp_path, "candidate.ivecs")


        config_file = os.path.join(exp_path,  "config/knnqdde_final_"+result_ends[i]+ ".json")
        with open(config_file, "w") as fout:
            json.dump(conf, fout, indent=4)

        subprocess.run([program_path, "-cf", config_file])
