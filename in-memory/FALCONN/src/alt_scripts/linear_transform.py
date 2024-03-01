import timeit


from sys import argv
import os, time
import helpers
import numpy as np
from fvecs import *

dim_list = {"audio": 192, "mnist":784, "enron":1369, "trevi":4096, 
    "gist":960, "glove":100, "deep":96, "sift":128}

def read_transform(dataset, qid, transform):
    count = dim_list[dataset] * (dim_list[dataset] + 1)
    offset = count * 4 * qid
    return fvecs_read(transform, count=count, offset=offset)


def run_gt(dataset, query_n, train, test, transform, ground_truth, new_query, dump, query_offset=0):
    query_stt = query_offset
    query_end = query_offset + query_n

    database = fvecs_read(train).T
    queryset = fvecs_read(test).T

    for query_cur in range(query_stt, query_end):
        dump_path = os.path.join(dump, str(query_cur) + ".fvecs")
        matrix = read_transform(dataset, query_cur, transform)
        transformed_data = matrix @ database
        transformed_query = matrix @ queryset[:, query_cur]

        e2distances_ = helpers.euclidean_distances(transformed_query, transformed_data.T)
        knn_id = np.argsort(e2distances_)[0,:200].reshape((1,200))
        

        with open(ground_truth, "a") as f:
            np.savetxt(f, knn_id, fmt="%10d")
        transformed_query = transformed_query.reshape((1, transformed_query.size))
        #to_fvecs(new_query, transformed_query, True)
        #to_fvecs(dump_path, transformed_data)

    
def run(dataset, dump):
    query_n = 100
    train = "/home/gtnetuser/alt/dataset/{}-train.fvecs".format(dataset)
    test = "/home/gtnetuser/alt/dataset/{}-test.fvecs".format(dataset)
    transform = "/home/gtnetuser/alt/half_kernels/{}.fvecs".format(dataset)
    ground_truth = "/home/gtnetuser/alt/ground_truth/{}.txt".format(dataset)
    new_query = "/home/gtnetuser/alt/dataset/{}-ALTtest.fvecs".format(dataset)

    run_gt(dataset, query_n, train, test, transform, ground_truth, new_query, dump)



if __name__ == "__main__":
    option = int(argv[1])

    datasets = [["audio", "enron", "gist", "deep"], ["mnist", "trevi", "glove", "sift"]]
    path = ["/media/mydrive/transformed_data", "/media/mydrive_2/transformed_data"]

    for dataset in datasets[option]:
        start_time = time.time()
        dump = os.path.join(path[option], dataset)
        try:
            os.makedirs(dump)
        except:
            pass
        run(dataset, dump)

        print("{} takes {} seconds.".format(dataset, time.time()-start_time))
