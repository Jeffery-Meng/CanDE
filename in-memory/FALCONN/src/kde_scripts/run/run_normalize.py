import os, subprocess

datasets = ["deep"]
#["enron", "gist", "trevi", "deep1M", "glove",  "sift", "audio", "mnist"]
dataset_path = "/media/mydrive/KroneckerRotation/data/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/normalize.py"
suffix = ["-train.fvecs", "-test.fvecs", "N-train.fvecs", "N-test.fvecs"]

for dataset in datasets:
    inputs = [os.path.join(dataset_path, dataset + x) for x in suffix]
    process = subprocess.Popen(["python3", script_path] + inputs)
    process.communicate()
