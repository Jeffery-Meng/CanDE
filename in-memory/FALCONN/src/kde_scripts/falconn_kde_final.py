import subprocess
import pathlib
import os, sys
import json
import numpy as np
import fvecs

from all_one import all_one

# Run ground truth for KDE.

datasets = ["audioN","treviN","gistN","enronN","gloveN","mnistN","siftN","deepN"]
tables = [12,12,12,29,10,16,10,12] # 12,12,29,10,16
#["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path_org = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
program_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn"
modes = ["knn and kde precomputed","knn and kde associative","knn and kde infer"] # ,
result_ends = ["precomputed","associative","infer"] # "infer","associative",


for j in range(len(datasets)):
    for i in range(len(modes)):
        dataset = datasets[j]
        table = tables[j]
        exp_path = os.path.join(exp_path_org,dataset)
        with open(os.path.join(exp_path, "config/kde_new.json")) as fin:
            conf = json.load(fin)
            
            # for key, val in conf.items():
            #     if isinstance(val, str):
            #             # print(val)
            #         val = val.replace("/media/gtnetuser/Elements/QDDE", "/media/gtnetuser/SSD_2TB_BEST/QDDE")
            #         val = val.replace("/media/mydrive/KroneckerRotation/data/", "/media/gtnetuser/SSD_2TB_ALPHA/data/")
            #         val = val.replace("/media/mydrive/distribution/","/media/gtnetuser/SSD_2TB_ALICE/distribution_new/")
            #         conf[key] = val
                
        with open(os.path.join(exp_path, "config/kde_new.json")) as fin:
            conf3 = json.load(fin)
        
        conf["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
        conf["gamma"] = conf3["gamma"]
        conf["save index"] = False
        conf["load distance"]=False
        conf["hash table parameters"][1]["l"] = table  
        conf["query mode"] = modes[i]
        conf["hash table width"] = 20
        # conf["ground truth file"] = os.path.join(exp_path, "kde_gt900.dvecs")
        
        del conf["hash table parameters"][0]
        conf["seed"] = 0x84f2c02b
        conf["knn filename"] = ""
        #os.path.join(exp_path,  "summary/knn_test_time_" + result_ends[i]+ ".ivecs")
        conf["kde filename"] = ""
        #os.path.join(exp_path,  "summary/kde_test_time_" + result_ends[i]+ ".fvecs")
        conf["accuracy filename"] = ""
        conf["accuracy binary filename"] = ""
        conf["cande table width"] = 16
        conf["compute ground truth"] = False
        del conf["gamma"][0:9]
        del conf["gamma"][1:]
        conf["result filename"] = os.path.join(exp_path,  "knn_kde_time_" + result_ends[i]+  ".txt")


        config_file = os.path.join(exp_path,  "config/knnkdei_final_"+result_ends[i]+ ".json")
        with open(config_file, "w") as fout:
            json.dump(conf, fout, indent=4)

        subprocess.run([program_path, "-cf", config_file])
