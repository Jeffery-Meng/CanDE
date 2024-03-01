import sys
sys.path.append("/media/mydrive/ann-codes/in-memory/aws_alsh-master/scripts")
from weight_gen import *
from fvecs import *
from plaintext import *
import numpy as np
from linear_scan_dist import linear_scan_ext
import os

NQuery = 100
NData = 10 ** 6
MaxK = 100 # used in SL-ALSH

pos = sys.argv.index("--weights")
weight_type = sys.argv[pos+1]

info_all = {
    "sift": {
        "data": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftC-train.fvecs",
        "query": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftC-test.fvecs",
        "weight": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/siftC2-{}W.fvecs".format(weight_type),
        "weight-plain": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/siftC2-{}W.txt".format(weight_type),
        "data-plain":  "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/siftC2-train.txt",
        "query-plain":  "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/siftC2-test.txt",
        "gt": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/siftC2-{}gt.ivecs".format(weight_type),
        "gt-plain": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/siftC2-{}gt.txt".format(weight_type),
        "dim": 128,
        "n": NData,
        "qn": NQuery
    }, "deep": {
        "data": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/deep-train.fvecs",
        "query": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/deep-test.fvecs",
        "weight": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/deep2-{}W.fvecs".format(weight_type),
        "weight-plain": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/deep2-{}W.txt".format(weight_type),
        "data-plain":  "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/deep2-train.txt",
        "query-plain":  "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/deep2-test.txt",
        "gt": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/deep2-{}gt.ivecs".format(weight_type),
        "gt-plain": "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/wl2/deep2-{}gt.txt".format(weight_type),
        "dim": 96,
        "n": NData,
        "qn": NQuery
    }
}

# transforms a given file to plaintext format
def file_transform(iptname, optname):
    data = fvecs_read(iptname)
    print_txt(optname, data)




dataset = sys.argv[1]
info = info_all[dataset]

if "--nooverwrite" in sys.argv and os.path.exists(info["gt-plain"]):
    exit(0)

# transform data query
print("reading data and query")
data = fvecs_read(info["data"])[:NData]
query = fvecs_read(info["query"])[:NQuery]

if "--noprintdata" not in sys.argv:
    print_txt(info["data-plain"], data, False)
    print_txt(info["query-plain"], query, False)

# weights
print("generating weights")
weights = weightGen(weight_type, NQuery, info["dim"])
saveWeight(weights, info["weight-plain"])
weights_nd = np.zeros((NQuery, info["dim"], info["dim"]))
for qid in range(NQuery):
    weights_nd[qid] = np.diag(np.sqrt(weights[qid]))
    # assert all weights are positive
    
weights_fv = weights_nd.reshape((NQuery*info["dim"], info["dim"]))
to_fvecs(info["weight"], weights_fv)

print("linear scan for ground truth")
# run ground truth, results are saved in functions 
linear_scan_ext(info, data, query, weights_nd)

