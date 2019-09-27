import numpy as np
from copy import deepcopy
from .num_diff import num_diff, h

class Layer:
    @classmethod
    def create(self,
            layer, 
            in_size, 
            out_size,
            *args,
            **kargs,
        ):
        if layer == 'id':
            return Id(in_size, out_size, *args)
        elif layer == 'affine':
            return Affine(in_size, out_size, *args)
        return Id(in_size, out_size)

class Id:
    def __init__(self, in_size, out_size, *args, **kargs):
        self.out_size = out_size
    def fp(self, x): #forward propagation
        return x
    def bp(self, propagated): #back propagation
        return propagated
    def update(self, propagated): # update weight of him self
        pass
    def num_diff_func(self):
        return self.fp
    def num_diff(self, inp):
        return num_diff(self.num_diff_func(), inp)

class Affine(Id):
    def __init__(self,
            in_size,
            out_size,
            *,
            weight = None,
            bias = None,
            learn_rate = 0.1
        ):
        self.in_size = in_size
        self.out_size = out_size
        self.learn_rate = learn_rate
        self.last_inp = None
        self.last_weight = None
        if(weight is None):
            self.weight = np.random.rand(self.out_size, self.in_size)
        else:
            self.weight = np.array(initial_weight)
        if(bias is None):
            self.bias = np.random.rand(self.out_size)
        else:
            self.bias = np.array(initial_bias)
    def fp(self, x):
        self.last_result = x @ self.weight.T + self.bias
        self.last_inp = x
        self.last_weight = self.weight
        return self.last_result
    def bp(self, prp):
        self.last_weight = self.weight
        diff = self.last_inp.reshape(self.last_inp.shape[0], 1) @ prp.reshape(1, prp.shape[0])
        self.weight = self.weight - diff.T * self.learn_rate
        return prp @ self.weight

    def num_diff_func(self):
        pass
    def num_diff(self, inp):
        copied = deepcopy(self.last_weight)
        calced = self.last_inp @ copied.T + self.bias
        ret = []
        for i in range(copied.shape[0]):
            each = []
            for j in range(copied.shape[1]):
                copied[i][j] += h
                calced = self.last_inp @ copied.T + self.bias
                each.append(((calced - self.last_result) / h)[i])
                copied[i][j] -= h
            ret.append(each)
        return np.array(ret).T @ inp
