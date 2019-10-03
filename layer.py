import numpy as np
from copy import deepcopy
from .num_diff import num_diff, h
from .update_strategy import UpdateStrategy
from .weight_distribution_strategy import WeightDitributionStrategy

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
            return Id(in_size, out_size, *args, **kargs)
        elif layer == 'affine':
            return Affine(in_size, out_size, *args, **kargs)
        return Id(in_size, out_size)

class Id:
    def __init__(self, in_size, out_size, *args, **kargs):
        self.out_size = out_size
    def fp(self, x): #forward propagation
        return x
    def bp(self, propagated): #back propagation
        return propagated
    def update(self):
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
            learn_rate = 0.1,
            update_strategy = None,
            weight_distribution_strategy = None,
        ):
        self.in_size = in_size
        self.out_size = out_size
        self.learn_rate = learn_rate
        self.last_inp = None
        self.last_weight = None
        self.learn_stack = []
        self.update_strategy = UpdateStrategy.create(update_strategy)
        self.weight_distribution_strategy = weight_distribution_strategy
        if(weight is None):
            self.weight = WeightDitributionStrategy.create(self.weight_distribution_strategy).distribute(self.in_size, self.out_size)
        else:
            self.weight = np.array(weight)
        if(bias is None):
            self.bias = np.random.rand(self.out_size) * 2. - 1.
        else:
            self.bias = np.array(bias)
    def fp(self, x):
        self.last_result = x @ self.weight + self.bias
        self.last_inp = x
        self.last_weight = self.weight
        return self.last_result
    def bp(self, prp):
        ip = self.last_inp
        pr = prp
        if(prp.ndim == 1): 
            ip = np.array([ip])
            pr = np.array([pr])
        self.learn_stack.append(self.update_strategy.calc(self, prp))
        return prp @ self.last_weight.T
    def update(self):
        self.weight += self.update_strategy.update(self)
    def num_diff_func(self):
        pass
    def num_diff(self, inp):
        copied = deepcopy(self.last_weight)
        calced = self.last_inp @ copied + self.bias
        ret = []
        for i in range(copied.shape[0]):
            each = []
            for j in range(copied.shape[1]):
                copied[i][j] += h
                calced = self.last_inp @ copied + self.bias
                each.append(((calced - self.last_result) / h)[i])
                copied[i][j] -= h
            ret.append(each)
        return np.array(ret) @ inp
