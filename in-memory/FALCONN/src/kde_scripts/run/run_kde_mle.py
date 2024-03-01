import os, subprocess, json

datasets = ["enronN", "gistN", "treviN", "gloveN",  "siftN", "mnistN"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_kde"

processes = []
for dataset in datasets:
    exp_folder = os.path.join(exp_path, dataset)
    with open(os.path.join(exp_folder, "kde.json")) as fin:
        conf = json.load(fin)
    with open(os.path.join(exp_folder, "ann_ps3.json")) as fin:
        conf2 = json.load(fin)
    conf["hash table parameters"] = conf2["hash table parameters"]
    conf["result filename"] = os.path.join(exp_folder, "kde_mle.txt")
    conf_path = os.path.join(exp_folder, "kde_mle.json")
    with open(conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)
    processes.append(subprocess.Popen([script_path, conf_path]))

for p in processes:
    p.communicate()
