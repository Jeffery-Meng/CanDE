# estimates KDE bandwidth from scott's rule of thumb.

from fvecs import *
import sys
import numpy as np
import json

def scott_rot(data_file):
    data = fvecs_read(data_file)
    n, dim = data.shape

    std = np.std(data, axis=0, ddof=1)
    maxstd, avgstd =  np.max(std), np.mean(std)
    scott = 1.06 * avgstd * n ** (-1 / (dim + 4))
    res = 1/ (2 * scott **2)
    scott2 = 1.06 * maxstd * n ** (-1 / (dim + 4))
    res2 = 1/ (2 * scott2 **2)
    return scott, res, scott2, res2

if __name__ == "__main__":
    with open(sys.argv[1]) as fin:
        conf = json.load(fin)
    print(scott_rot(conf["data filename"]))