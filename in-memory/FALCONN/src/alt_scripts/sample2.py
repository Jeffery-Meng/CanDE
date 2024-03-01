from sys import argv
import os
import numpy as np
import shutil
import struct



rng = np.random.default_rng()

datasets = ["audio", "MNIST", "enron", "trevi", "glove"]
DIR = "/mnt/nvme0n1p1/distribution/ann-codes/in-memory/EXPERIMENTS"

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def to_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', y.size)	
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def combine(dataset):
    train_path = os.path.join(DIR, dataset, dataset+"-train.fvecs")
    test_path = os.path.join(DIR, dataset, dataset+"-test.fvecs")
    
    train = fvecs_read(train_path)
    test = fvecs_read(test_path)

    train_a = train.shape[0]
    idxes = rng.choice(range(train_a), 800, replace=False)
    remains = np.setdiff1d(range(train_a), idxes)
    train_n = train[remains, :]
    train_r = train[idxes, :]
    print(test.shape, train_r.shape)
    test_n = np.concatenate((test, train_r))

    to_fvecs("dataset/" + dataset + "-train.fvecs", train_n)
    to_fvecs("dataset/" + dataset + "-test.fvecs", test_n)

for d in datasets:
    combine(d)



