import struct
import numpy as np

def to_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', y.size)	
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def to_ifvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', y.size)	
            fp.write(d)
            for x in y:
                a = struct.pack('I', int(x))
                fp.write(a)

def fvecs_read(filename, c_contiguous=True, count=-1, offset=0):
    # count is the number of data items to read (including the leading SIZE in fvecs format)
    fv = np.fromfile(filename, dtype=np.float32, count=count, offset=offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def ivecs_read(filename):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    return fv