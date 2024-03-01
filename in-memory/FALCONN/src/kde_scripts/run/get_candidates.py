import subprocess
import pathlib
import os, sys
import json
import numpy as np

# Run ground truth for KDE.

datasets = ["gist"]
#["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path_org = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/EXPERIMENTS/"
program_path = "/media/gtnetuser/SSD_2TB_ALICE/distribution_new/ann-codes/in-memory/FALCONN/build/falconn"


for j in range(len(datasets)):
    dataset = datasets[j]
    exp_path = os.path.join(exp_path_org,dataset)
    with open(os.path.join(exp_path, "config/knnqdde_final_precomputed.json")) as fin:
        conf = json.load(fin)
                     
    conf["index path"] = "/media/gtnetuser/SSD_2TB_ALPHA/QDDE/{}/index/".format(dataset)
    conf["printing mode"]["bucket"] = False 
    conf["query mode"] = "knn candidates"
    conf["candidate filename"] = os.path.join(exp_path,"candidates","dedup_")
    if not os.path.exists(os.path.join(exp_path,"candidates")):
        os.mkdir(os.path.join(exp_path,"candidates"))
    
    # conf["ground truth file"] = os.path.join(exp_path, "kde_gt900.dvecs")

    config_file = os.path.join(exp_path,  "config/knnqdde_print_candiates.json")
    with open(config_file, "w") as fout:
        json.dump(conf, fout, indent=4)

    subprocess.run([program_path, "-cf", config_file])
