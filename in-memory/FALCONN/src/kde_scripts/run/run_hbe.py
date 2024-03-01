import os, subprocess

datasets = ["audioN", "gistN", "treviN", "gloveN", "mnistN", "deepN", "siftN", "enronN"]
dataset_path = "/media/mydrive/KroneckerRotation/data/"
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/falconn_run_hbe.py"

bin_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build"
subprocess.run(["make", "-C", bin_path, "hbe"])

processes = []
for dataset in datasets:
    processes.append(subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset)]))

for p in processes:
    p.communicate()
