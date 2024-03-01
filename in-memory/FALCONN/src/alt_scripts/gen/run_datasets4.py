import os, subprocess, json

datasets = ["audio", "mnist"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
dataset_path = "/media/mydrive/KroneckerRotation/data/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/count_candidates"
processes = []

for dataset in datasets:
    exp_data_path = os.path.join(exp_path, dataset)
    with open(os.path.join(exp_data_path, "ann_mle.json")) as fin:
        conf = json.load(fin)
    conf["result filename"] = os.path.join(exp_data_path, "knn_candidate_count.txt")
    conf_path = os.path.join(exp_data_path, "ann_cc.json")
    with open(conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)
    process = subprocess.Popen([script_path, conf_path])
    process.communicate()

