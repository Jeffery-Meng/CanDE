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
bin_path.mkdir(parents=True, exist_ok=True)
build_cmd = ["make", "-C", str(bin_path),  "-j"]
subprocess.run(build_cmd)

auto_prepare_cmd = ["python3", os.path.join(this_path, "auto_prepare.py"), dataset_name,
    exp_path, data_file, query_file]

if "--iskde" in sys.argv:
    auto_prepare_cmd.append("--iskde")
subprocess.run(auto_prepare_cmd)

exp_path = os.path.join(exp_path, dataset_name)
auto_param_cmd = ["python3", os.path.join(this_path, "command_console.py")]
auto_param_line = os.path.join(this_path, "auto_param.py") + " "
auto_param_line += "parameters "
auto_param_line += exp_path
auto_param_line += "\n"
if "--maxprocess" in sys.argv:
    idx = sys.argv.index("--maxprocess")
    auto_param_cmd.extend(sys.argv[idx:idx+2])
subprocess.run(auto_param_cmd, input=auto_param_line.encode())


summary_path = os.path.join(exp_path, "summary")
# Delete old log files
for f in glob.glob(os.path.join(summary_path, "w*.txt")):
    os.remove(f)

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

parameter_file = os.path.join(summary_path, "parameters.txt")
with open(parameter_file, 'w') as fout:
    for param in param_list:
        fout.write("{}\t{}\t{}\t{}\n".format(*param))

config_file = os.path.join(exp_path, "ann.json")
with open(config_file) as fin:
    config = json.load(fin)

config["hash table parameters"] = [
        {
            "k": m,
            "l": 30,
            "bucket width": w
        }
    ]
config["query mode"] = "knn candidates"
with open(config_file, "w") as fout:
    json.dump(config, fout, indent=4)