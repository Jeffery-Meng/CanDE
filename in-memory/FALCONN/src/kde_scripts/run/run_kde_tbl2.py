import os, subprocess, json, sys
import numpy
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["enronN", "gistN", "treviN", "gloveN",  "siftN", "mnistN", "deepN"]
nb_range = [1, 4, 20]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_kde_tbl"

processes = []
for dataset in datasets:
    exp_folder = os.path.join(exp_path, dataset)
    for idx, n_bins in enumerate(numpy.logspace(*nb_range)):
        n_bins = int(n_bins)
        conf_path = os.path.join(exp_folder, "config", "kde_hist.json")
        with open(conf_path) as fin:
            conf = json.load(fin)
        conf["histogram bins"] = conf["histogram bins"][4*idx: 4*(idx+1)]
        conf["histogram filename"] =  os.path.join(exp_folder, "summary",
                "kde_histogram_{}.dvecs".format(n_bins))
        conf["result filename"] = os.path.join(exp_folder, "summary",
                "kde_tbl_mres_{}.txt".format(n_bins))
        conf_path = os.path.join(exp_folder, "config", "kde_tbl_{}.json".format(n_bins))
        
        with open(conf_path, "w") as fout:
            json.dump(conf, fout, indent=4)
        processes.append(subprocess.Popen([script_path, conf_path]))

    for p in processes:
        p.communicate()
