import os, subprocess

datasets = ["deepN"]
#["enronN", "mnistN", "treviN", "deepN", "gloveN",  "siftN", "audioN", "gistN"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/"
dataset_path = "/media/mydrive/KroneckerRotation/data/"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/falconn_param_tuning.py"
processes = []

for dataset in datasets:
    data_name = os.path.join(dataset_path, "{}-train.fvecs".format(dataset))
    query_name = os.path.join(dataset_path, "{}-test.fvecs".format(dataset))
    processes.append(subprocess.Popen(["python3", script_path, dataset, exp_path, data_name, query_name,
            "--targetrecall", "0.6"]))

for p in processes:
    p.communicate()
