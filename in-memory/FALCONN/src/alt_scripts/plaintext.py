""" This file contains I/O functions for legacy data formats 
used in Weighted L2 experiments
"""

import numpy as np
import os

# print np array to plain text file
def print_txt(filename, arr, overwrite=True):
    if os.path.exists(filename) and not overwrite:
        return
    line_num = arr.shape[0]
    with open(filename, "w") as fout:
        for lid in range(line_num):
            fout.write(str(lid) + " ")
            fout.write(" ".join([str(x) for x in arr[lid]]))
            fout.write("\n")

def print_gt(filename, ids, dists):
    nq, max_k = ids.shape
    assert(ids.shape == dists.shape)
    with open(filename, "w") as fout:
        fout.write("{} {}\n".format(nq, max_k))
        for lid in range(nq):
            for pair in zip(ids[lid], dists[lid]):
                fout.write("{} {} ".format(*pair))
            fout.write("\n")


if __name__ == "__main__":
    outputpath = "/media/mydrive/ann-codes/in-memory/aws_alsh-master/tests/testdata/"
    print_txt(outputpath+"data.txt", np.array([[1,2,3], [2,3,4], [2,3,4]]))
    print_gt(outputpath+"gt.txt", np.array([[1,2,3], [2,3,4], [2,3,4]]), np.array([[13,23,33], [23,33,43], [23,34,44]]))

