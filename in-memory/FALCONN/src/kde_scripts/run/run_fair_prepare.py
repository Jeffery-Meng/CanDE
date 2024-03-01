import os, subprocess
import json

datasets = ["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/falconn_fair.py"
processes = []

for dataset in datasets:
    processes.append(subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset), "500"]))
for process in processes:
    process.communicate()

