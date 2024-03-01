from fvecs import *
from plaintext import *
from sys import argv

iptname = argv[1]
optname = argv[2]

data = fvecs_read(iptname)
print_txt(optname, data)