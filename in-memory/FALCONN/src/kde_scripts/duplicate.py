import sys
import json, os.path
from fvecs import *
import numpy as np

CONFIG_PATH = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/kdd99.json"

with open(CONFIG_PATH) as fin:
    config = json.load(fin)

data = fvecs_read(config["data filename"])
n, dim = data.shape
assert(n > dim)  # check data is not inverted
query = fvecs_read(config["query filename"])
nq, dimq = query.shape
assert(dim == dimq)

dist = fvecs_read(config["distance file"])
    
max_cnt = 0
for q in range(nq):
    cnt = 0
    for i in range(n):
        if dist[q, i] < 1e-5:
            cnt += 1
    if cnt > max_cnt:
        max_cnt = cnt

print(max_cnt)
exit(1)


with open(sys.argv[1], "w") as fout:
    json.dump(config, fout, indent=4)
