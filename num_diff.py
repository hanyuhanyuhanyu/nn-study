import numpy as np

h = 1e-7
def error_num_diff(f, result, *_):
    common = f(result, *_)
    ret = []
    for i in range(result.shape[0]):
        result[i] += h
        ret.append(((f(result, *_) - common) / h))
        result[i] -= h
    return np.array(ret)

def num_diff(f, result):
    common = f(result)
    ret = []
    for i in range(result.shape[0]):
        target = result[i] + h
        ret.append((f(target) - common[i]) / h)
    return np.array(ret)

