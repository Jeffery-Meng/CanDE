from fvecs import *
import sys

data = ivecs_read_non_uniform(sys.argv[1])
with open(sys.argv[2], "w") as fout:
    for row in data:
        for val in row:
            fout.write(str(val) + "\t")
        fout.write("\n")