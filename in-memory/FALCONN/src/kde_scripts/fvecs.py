import struct
import numpy as np

def to_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        if len(data.shape) == 1:
            # 1D vector
            d = struct.pack('I', data.size)	
            fp.write(d)
            for x in data:
                a = struct.pack('f', x)
                fp.write(a)
        else:
            for y in data:
                d = struct.pack('I', y.size)	
                fp.write(d)
                for x in y:
                    a = struct.pack('f', x)
                    fp.write(a)

def to_ivecs(filename, data):
    with open(filename, 'wb') as fp:
        print(data.shape)
        if len(data.shape) == 1:
            # 1D vector
            d = struct.pack('I', data.size)	
            fp.write(d)
            for x in data:
                a = struct.pack('I', x)
                fp.write(a)
        else:
            for y in data:
                d = struct.pack('I', y.size)	
                fp.write(d)
                for x in y:
                    a = struct.pack('I', x)
                    fp.write(a)

def fvecs_read(filename, c_contiguous=True, count=-1, offset=0):
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
    
def ivecs_read_non_uniform(filename):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    cnt = 0
    result = []
    while cnt < fv.size:
        sz = fv[cnt]
        assert cnt + sz + 1 <= fv.size
        result.append(fv[cnt+1:cnt+sz+1])
        cnt += sz + 1
    return result

def fvecs_read_non_uniform(filename):
    fv = np.fromfile(filename, dtype=np.float32)
    fv_int = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    cnt = 0
    result = []
    while cnt < fv.size:
        sz = fv_int[cnt]
        assert cnt + sz + 1 <= fv.size
        result.append(fv[cnt+1:cnt+sz+1])
        cnt += sz + 1
    return result

def dvecs_read(filename, c_contiguous=True, count=-1, offset=0):
    fv = np.fromfile(filename, dtype=np.float32, count=count, offset=offset)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim * 2)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv.view(np.float64)