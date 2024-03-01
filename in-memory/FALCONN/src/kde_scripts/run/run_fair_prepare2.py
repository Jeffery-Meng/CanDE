import os, subprocess
import json

configs = ["fairnn_37000.json", "fairnn_57000.json", "fairnn_87000.json"]
exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/audio"
script_path = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn"
processes = []

for conf in configs:
    processes.append(subprocess.Popen([script_path, "-cf", os.path.join(exp_path, conf)]))
for process in processes:
    process.communicate()

