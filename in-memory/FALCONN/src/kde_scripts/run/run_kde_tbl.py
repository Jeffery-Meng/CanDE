import os, subprocess, json, sys
import numpy
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import fvecs

datasets = ["enronN", "gistN", "treviN", "gloveN",  "siftN", "mnistN", "deepN"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_disthist_tbl2"
script_path2 = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn_kde_tbl"
nb_range = [1, 4, 20]

processes = []
for dataset in datasets:
    if dataset != "deepN":
        continue
    exp_folder = os.path.join(exp_path, dataset)
    with open(os.path.join(exp_folder, "kde_mle.json")) as fin:
        conf = json.load(fin)
    distances = fvecs.fvecs_read(conf["distance file"])
    lb = float(numpy.min(distances))
    ub = float(numpy.quantile(distances, 0.99))
    del distances
    histogram_bins = []
    for n_bins in numpy.logspace(*nb_range):
        n_bins = int(n_bins)
        step = (ub - lb) / n_bins
        histogram_bins.extend([lb, ub, step, n_bins])

    conf["histogram bins"] = histogram_bins
    conf["result filename"] = os.path.join(exp_folder, "summary",
            "histogram_time_")
    conf["row id filename"] = os.path.join(exp_folder, "summary",
            "histogram_mres_")
    conf["histogram filename"] =  os.path.join(exp_folder, "summary",
            "kde_histogram_")
    conf_path = os.path.join(exp_folder, "config", "kde_hist.json")
    with open(conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)
    processes.append(subprocess.Popen([script_path, conf_path]))

for p in processes:
        p.communicate()

for dataset in datasets:
    processes = []
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
        processes.append(subprocess.Popen([script_path2, conf_path]))

    for p in processes:
        p.communicate()