import subprocess
import pathlib
import os, sys, glob
import json

dataset_name = sys.argv[1]
exp_path = sys.argv[2]
data_file = sys.argv[3]
query_file = sys.argv[4]

idx = sys.argv.index("--targetrecall")
targ_recall = float(sys.argv[idx+1])

this_path = pathlib.Path(__file__).parent.resolve()

bin_path = this_path.parent.parent / "build"

exp_path = os.path.join(exp_path, dataset_name)

with open(os.path.join(exp_path, "ann.json")) as fin:
    conf = json.load(fin)
conf["query variant"] = "radius r"
conf["query mode"] = "knn recall"
radius_list = [37000, 57000, 87000]
for r in radius_list:
    conf["nn radius"] = r
    conf_path = os.path.join(exp_path, "fairnn_{}.json".format(r))
    with open(conf_path, "w") as fout:
        json.dump(conf, fout, indent=4)

    auto_param_cmd = ["python3", os.path.join(this_path, "command_console.py")]
    auto_param_line = os.path.join(this_path, "auto_param.py") + " "
    auto_param_line += "parameters "
    param_list = {"path": exp_path, "cf_path": conf_path, "ll":6, "lu":16, "du":2.5, "dn":40}
    auto_param_line += json.dumps(param_list)
    auto_param_line += "\n"
    if "--maxprocess" in sys.argv:
        idx = sys.argv.index("--maxprocess")
        auto_param_cmd.extend(sys.argv[idx:idx+2])
    subprocess.run(auto_param_cmd, input=auto_param_line.encode())


    summary_path = os.path.join(exp_path, "summary")
    new_path = pathlib.Path(exp_path) / "summary" / str(r)
    new_path.mkdir(exist_ok=True, parents=True)
    # Delete old log files
    for f in glob.glob(os.path.join(summary_path, "w*.txt")):
        os.rename(f, str(new_path / f.split("/")[-1]))

    param_list = []
    for file in os.listdir(summary_path):
        if file[0] != "w":
            continue
        with open(os.path.join(summary_path, file)) as f:
            f.readline()
            try:
                l, m, w, recall, ratio, ctime = (float(x) for x in f.readline().split())
            except:
                continue
        if recall < targ_recall:
            continue
        param_list.append((ratio, m, w, recall))

    param_list.sort()
    ratio, m, w, recall = param_list[0]
    m = int(m)
    print(m, w)

    parameter_file = os.path.join(summary_path, "parameters_{}.txt".format(r))
    with open(parameter_file, 'w') as fout:
        for param in param_list:
            fout.write("{}\t{}\t{}\t{}\n".format(*param))

    config_file = os.path.join(exp_path, "fairnn_{}.json".format(r))
    with open(config_file) as fin:
        config = json.load(fin)

    config["hash table parameters"] = [
            {
                "k": m,
                "l": 500,
                "bucket width": w
            }
        ]
    config["query mode"] = "knn duplicate candidates"
    with open(config_file, "w") as fout:
        json.dump(config, fout, indent=4)