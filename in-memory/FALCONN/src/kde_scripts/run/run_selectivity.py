import os, subprocess, json

#datasets = ["audio", "enron", "gist", "trevi", "glove",  "sift", "mnist", "deep"]
datasets = ["audioN", "deepN", "enronN", "gistN", "treviN", "gloveN",  "siftN", "mnistN"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_selectivity"

processes = []
for dataset in datasets:
    exp_folder = os.path.join(exp_path, dataset)
  #  processes.append(subprocess.Popen([script_path, os.path.join(exp_folder, "ann_ps3b.json")]))

for p in processes:
    p.communicate()

with open(os.path.join(exp_path, "selectivity.txt"), "w") as fout:
    for dataset in datasets:
        exp_folder = os.path.join(exp_path, dataset)

        with open(os.path.join(exp_folder, "ann_ps3b.json")) as fin:
            conf = json.load(fin)
            num_hash_functions = conf["hash table parameters"][1]["l"]
        with open(os.path.join(exp_folder, "summary/selectivity.txt")) as fin:
            selectivity = float(fin.readline())
        fout.write("{}\t{}\t{}\n".format(dataset, num_hash_functions, selectivity))
        
