import os, subprocess

datasets = ["enronN", "mnistN"]
#["enronN", "mnistN", "treviN", "deepN", "gloveN",  "siftN", "audioN", "gistN"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
dataset_path = "/media/mydrive/KroneckerRotation/data/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/falconn_qde_printing.py"
processes = []

for dataset in datasets:
    processes.append(subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset)]))

for p in processes:
    p.communicate()
