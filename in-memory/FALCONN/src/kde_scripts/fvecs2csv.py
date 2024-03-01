from fvecs import *
import sys
import os
import numpy as np

def fvecs_to_csv(in_path, out_path):
    data = fvecs_read(in_path)
    np.savetxt(out_path, data, delimiter=",", fmt='%.8e')

if __name__ == "__main__":
    dpath = "/media/mydrive/KroneckerRotation/data"
    datasets = ["audioN", "enronN", "mnistN", "treviN", "gistN", "gloveN", "deepN", "siftN"]
    opath = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/datasets"
    train_test = ["train", "test"]
    for data in datasets:
        for tq in train_test:
            in_path = os.path.join(dpath, "{}-{}.fvecs".format(data, tq))
            out_path = os.path.join(opath, "{}-{}.csv".format(data, tq))
            fvecs_to_csv(in_path, out_path)

