import timeit


from sys import argv
import os, time
import numpy as np
from fvecs import *

subspace_gt = ivecs_read("/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftS-gt.ivecs")
subspace_l2 = ivecs_read("/media/mydrive/ann-codes/in-memory/EXPERIMENTS/dataset/siftC-gt.ivecs")

def run(qid):
    s3 = set(subspace_gt[qid, :100])
    s1 = set(subspace_l2[qid*2, :100])
    s2 = set(subspace_l2[qid*2+1, :100])

    return len(s1.union(s2).intersection(s3))

sum = 0
for qid in range(100):
    sum += run(qid)

print(sum / 100)