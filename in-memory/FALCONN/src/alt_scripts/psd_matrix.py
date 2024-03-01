from sys import argv
import numpy as np
from fvecs import to_fvecs

DIR = "/media/mydrive_2/gaussian ensemble/half_kernels/"
num_queries = 1000
LARGE = (0.8, 1)
SMALL = (0.2, 0.4)
rng = np.random.default_rng()

def generate_one(large, small):
    """ Returns Half kernel - sqrt(LAMBDA) * rotation """
    dim = large + small
    lamb = np.zeros((dim, dim))
    
    for d in range(large):
        lamb[d, d] = rng.uniform(*LARGE) ** 0.5
    for d in range(large, dim):
        lamb[d, d] = rng.uniform(*SMALL) ** 0.5

    gauss = rng.normal(size=(dim, dim))
    rotation, _ = np.linalg.qr(gauss)
    return np.dot(lamb, rotation)

def generate_dataset(dataset, dim):

    larges = np.ceil(np.linspace(1, dim, num_queries)).astype('int')
    smalls = dim - larges

    result = np.empty((num_queries, dim, dim), dtype=np.float32)
    for i, para in enumerate(zip(larges, smalls)):
        result[i] = generate_one(*para)

    path = DIR + dataset + ".fvecs"
    result = result.reshape((num_queries*dim, dim))
    to_fvecs(path, result)


with open("datasets.txt", "r") as fin:
    for line in fin.readlines():
        dataset, dim = tuple(line.split())
        dim = int(dim)
        generate_dataset(dataset, dim)