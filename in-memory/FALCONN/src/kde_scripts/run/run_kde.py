import os, subprocess

datasets = ["gistN", "gloveN"] # ["audioN" , "gistN", "treviN", "gloveN", "mnistN", "deepN", "siftN", "enronN"]
dataset_path = "/media/mydrive/KroneckerRotation/data/"
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/falconn_run_kde6.py"

bin_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build"
subprocess.run(["make", "-C", bin_path, "kde_gt", "falconn_kde", "falconn_disthist_tbl2", "falconn_kde_v3",
    "kde_sample", "falconn_disthist_sketch", "hbe"])

processes = []
for dataset in datasets:
    processes.append(subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset)]))

for p in processes:
    p.communicate()
