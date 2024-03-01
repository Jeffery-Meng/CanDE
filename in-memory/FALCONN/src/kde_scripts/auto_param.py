TABLE_NUM = 5
HFUNC_NUM_MIN = 4
HFUNC_NUM_MAX = 8

DIST_RATIO_MIN = 1.0
DIST_RATIO_MAX = 3.0
DIST_RATIO_NUM = 30

NEIGHBOR_NUM = 100
#CONFIG_PATH = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/audio/audio.json"
#CONFIG_DUMP_PATH = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/audio/config"
FALCONN_BIN_PATH = "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/build/falconn"

from fvecs import *
import json
import os.path 

def average_distance(dist_file):
    distances = fvecs_read(dist_file)
    distances.sort(axis=1)
    near_distances = np.mean(distances[:, :NEIGHBOR_NUM], axis=1)
    return np.mean(near_distances), np.std(near_distances)

def print_distance(dist_file):
    distances = fvecs_read(dist_file)
    distances.sort(axis=1)
    near_distances = np.mean(distances[:, :NEIGHBOR_NUM], axis=1)
    for id, val in enumerate(near_distances):
        if val < 1e-5:
            print(distances[id, :NEIGHBOR_NUM])

def parameters(path, cf_path = None,  ll = HFUNC_NUM_MIN, lu = HFUNC_NUM_MAX, dl = DIST_RATIO_MIN,
    du = DIST_RATIO_MAX, dn = DIST_RATIO_NUM):
    if cf_path is None:
        cf_path = os.path.join(path, "ann.json")
    CONFIG_DUMP_PATH = os.path.join(path, "config")
    with open(cf_path) as fin:
        config = json.load(fin)
    
    avg_dist, std_dist = average_distance(config["distance file"])
    if (std_dist > avg_dist * 0.1):
        print("Warning: NN distances vary too much among queries")
    hash_params = config["hash table parameters"][0]

    for h_func in range(ll, lu + 1):
        for dist_ratio in np.linspace(dl, du, dn):
            width = dist_ratio * avg_dist
            hash_params["l"] = TABLE_NUM
            hash_params["k"] = h_func
            hash_params["bucket width"] = width

            config["hash table parameters"][0] = hash_params
            json_name = "k_{}_w_{}.json".format(h_func, width)
            json_name = os.path.join(CONFIG_DUMP_PATH, json_name)
            with open(json_name, "w") as fout:
                json.dump(config, fout, indent=4)
            yield [FALCONN_BIN_PATH, "-cf", json_name]



if __name__ == "__main__":
#    with open(CONFIG_PATH) as fin:
#        config = json.load(fin)
    
#    print(average_distance(config["distance file"]))
    pass
