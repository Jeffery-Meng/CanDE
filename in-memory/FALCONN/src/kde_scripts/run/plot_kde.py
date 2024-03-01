import os, subprocess

datasets = ["enronN", "gistN", "treviN", "deepN", "gloveN",  "siftN", "audioN", "mnistN"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
dataset_path = "/media/mydrive/KroneckerRotation/data/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/kde_figures.py"
processes = []

for dataset in datasets:
    processes.append(subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset)]))
for process in processes:
    process.communicate()

