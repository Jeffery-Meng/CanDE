import numpy as np
from fvecs import *
from sys import argv
from scipy.linalg import null_space

def affine_space(A):
    '''Input: a matrix A consists of vectors in space
    Compute a matrix C and vector d such that affine-span(A)  = {Cx = d}
    Output: C^TC, a0, to be used in ONIAK experiments '''
    a0 = A[0]
    B = A[1:] - a0    
    C = null_space(B)  # contains all vectors such that BC = 0
    #print(C.T @ C)
    return C.T, a0

# l2 distance = ||C.T*C*(x - a0)||2 

dataset = argv[1]
dim_as = [2,4,8,16,32,64]
info_all = {
    "sift": {"dim": 128, "n": 5*10**7, "qn": 10000, 
    "queryfile": "/home/gtnetuser/alt/dataset/siftC-test.fvecs"},
    "deep" : {"dim": 96, "n": 5*10**7, "qn": 10000, 
    "queryfile": "/home/gtnetuser/alt/dataset/deep-test.fvecs"},
    "audio" : {"dim": 192, "n": 5*10**7, "qn": 200, 
    "queryfile": "/home/gtnetuser/alt/dataset/audio-testC.fvecs"}
}
info = info_all[dataset]
dim = info["dim"]
qn = 100
#dim_a = 2 # Real dimension is minus 1
savepath = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/subspace/"
queries = fvecs_read(info["queryfile"])
rng = np.random.default_rng()

def run(dim_a):
    subspace_queries = np.zeros((qn, dim))
    subspace_kernels = np.zeros((qn*dim, dim))
    for qid in range(qn):
        selected = rng.permutation(info["qn"])[:dim_a]
        c, a = affine_space(queries[selected])
        subspace_queries[qid] = a
        subspace_kernels[qid*dim:(qid+1)*dim] = c.T@c

    to_fvecs(savepath + "{}{}-kernels.fvecs".format(dataset, dim_a), subspace_kernels)
    to_fvecs(savepath + "{}{}-queries.fvecs".format(dataset, dim_a), subspace_queries)

for dim_a in dim_as:
    run(dim_a)