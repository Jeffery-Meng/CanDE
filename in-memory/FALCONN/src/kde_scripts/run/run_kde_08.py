import os, subprocess

#datasets = #["audioN", "enronN", "gistN", "treviN", "gloveN",  "siftN", "mnistN", "deepN"]
datasets = ["audio", "enron", "gist", "trevi", "glove",  "sift", "mnist", "deep"]
dataset_path = "/media/mydrive/KroneckerRotation/data/"
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/falconn_qde_printing2.py"

bin_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build"
subprocess.run(["make", "-C", bin_path, "kde_gt", "falconn_kde", "falconn_disthist_tbl2", "falconn_kde_v3"])

processes = []
for dataset in datasets:
    processes.append(subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset), "--targetrecall", "0.8"]))

for p in processes:
    p.communicate()
