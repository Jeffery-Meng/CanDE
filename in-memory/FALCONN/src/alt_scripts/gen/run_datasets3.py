import os, subprocess

datasets = ["enron", "gist", "trevi", "deep", "glove",  "sift", "audio", "mnist"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
dataset_path = "/media/mydrive/KroneckerRotation/data/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/mre_figures4.py"
processes = []

for dataset in datasets:
    process = subprocess.Popen(["python3", script_path, os.path.join(exp_path, dataset)])
    process.communicate()

