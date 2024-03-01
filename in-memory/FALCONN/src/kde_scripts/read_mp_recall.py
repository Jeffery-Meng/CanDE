import sys
from fvecs import *

radius = 8.0 / 11.5288
idx = int(radius / 0.0001)

recalls = fvecs_read_non_uniform(sys.argv[1])[1]
r = recalls[idx]
print(r)
print( 1  - ( 1- r) ** 20)
print( 1  - ( 1- r) ** 100)