import os, subprocess, json, numpy

#datasets = ["audio", "enron", "gist", "trevi", "glove",  "sift", "mnist", "deep"]
datasets = ["audio", "enron", "gist", "trevi", "glove",  "sift", "mnist", "deep"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
recall_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_recall_table"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_selectivity"

processes = []
for dataset in datasets:
    exp_folder = os.path.join(exp_path, dataset)
    with open(os.path.join(exp_folder, "ann_cd.json")) as fin:
        conf = json.load(fin)
    conf["hash table parameters"].append(conf["hash table parameters"][0].copy())
    conf["hash table parameters"][1]["l"] = 50
    conf_path = os.path.join(exp_folder, "config/ann_recall.json")
    with open(conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)
#    processes.append(subprocess.Popen([recall_path, conf_path]))
for p in processes:
    p.communicate()

processes = []
for dataset in datasets:
    exp_folder = os.path.join(exp_path, dataset)
    recall_summary_path = os.path.join(exp_folder , "summary" , "recalls.txt")

    with open(recall_summary_path) as fin:
        recalls = [float(x) for x in fin.readline().split()]
    with open(os.path.join(exp_folder, "ann_ps3.json")) as fin:
        config = json.load(fin)
    active_tables = numpy.searchsorted(recalls, 0.8) + 1
    if active_tables > config["hash table parameters"][0]["l"]:
        active_tables = config["hash table parameters"][0]["l"]
    config["hash table parameters"][1]["l"] = int(active_tables)
    with open(os.path.join(exp_folder, "ann_ps3b.json"), "w") as fout:
        json.dump(config, fout, indent=4)
 #   processes.append(subprocess.Popen([script_path, os.path.join(exp_folder, "ann_ps3b.json")]))

for p in processes:
    p.communicate()

with open(os.path.join(exp_path, "selectivity_original.txt"), "w") as fout:
    for dataset in datasets:
        exp_folder = os.path.join(exp_path, dataset)

        with open(os.path.join(exp_folder, "ann_ps3b.json")) as fin:
            conf = json.load(fin)
            num_hash_functions = conf["hash table parameters"][1]["l"]
        with open(os.path.join(exp_folder, "summary/selectivity.txt")) as fin:
            selectivity = float(fin.readline())
        fout.write("{}\t{}\t{}\n".format(dataset, num_hash_functions, selectivity))
        
