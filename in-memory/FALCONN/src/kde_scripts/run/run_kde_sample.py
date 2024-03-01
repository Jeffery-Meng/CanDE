import os, subprocess, ast, json

dataset_path = "/media/mydrive/KroneckerRotation/data/"
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/kde_sample"

bin_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build"
subprocess.run(["make", "-C", bin_path, "kde_gt", "falconn_kde", "falconn_disthist_tbl2", "falconn_kde_v3", 
        "kde_sample"])

with open(os.path.join(exp_path, "kde_results2.txt")) as fin:
    errors = [ast.literal_eval(x) for x in fin.readlines()]

processes = []
for error_entry in errors:
    dataset = error_entry[0]
    cur_path = os.path.join(exp_path, dataset)
    with open(os.path.join(cur_path, "kde_mle.json")) as fin:
        conf = json.load(fin)
    
    conf["target error"] = error_entry[2]
    conf["error range"] = 0.1 * conf["target error"]
    conf["dataset name"] = dataset
    conf["result filename"] = os.path.join(cur_path, "kde_sample.txt")
    config_path = os.path.join(cur_path, "kde_sample.json")
    with open(config_path, "w") as fout:
        json.dump(conf, fout, indent=4)
    processes.append(subprocess.Popen([script_path, config_path]))


for p in processes:
    p.communicate()


for error_entry in errors:
    dataset = error_entry[0]
    cur_path = os.path.join(exp_path, dataset)
    with open(os.path.join(cur_path, "kde_mle.json")) as fin:
        conf = json.load(fin)
    
    conf["target error"] = error_entry[3]
    conf["error range"] = 0.1 * conf["target error"]
    conf["dataset name"] = dataset
    conf["result filename"] = os.path.join(cur_path, "kde_sample_tbl.txt")
    config_path = os.path.join(cur_path, "kde_sample.json")
    with open(config_path, "w") as fout:
        json.dump(conf, fout, indent=4)
    processes.append(subprocess.Popen([script_path, config_path]))

for p in processes:
    p.communicate()