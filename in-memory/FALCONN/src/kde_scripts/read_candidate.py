import sys
import json, os.path, struct
from fvecs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pathlib

def load_candidate(query, config):
    l = 17
    with open(config) as fin:
        config = json.load(fin)
    candidate_file = config["candidate filename"] + "_q_{}.bin".format(query)
    with open(candidate_file, "rb") as fin:
        candfile = fin.read()

    mode = struct.unpack("@B", candfile[:1])
    mode = mode[0]
    assert(mode == 0b11100000)
    length = struct.unpack("@i", candfile[1:5])
    length = length[0]
    candidates = []
    candidate_tables = []
    pointer = 5
    l_max = config["hash table parameters"][0]["l"]
    for i in range(l_max):
        candidate_tables.append([])
    for cand in range(length):
        id, table, bucket = struct.unpack("@iii", candfile[pointer: pointer + 12])
        pointer += 12
        if table < l  and table >= 0:
            candidates.append(id)
        candidate_tables[table].append(id)
    return candidate_tables
