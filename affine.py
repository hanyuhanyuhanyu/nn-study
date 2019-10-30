import numpy as np
from .layer import Layer
from .weight_initializer import WeightInitializer
from .bias_initializer import BiasInitializer
from .update_strategy import UpdateStrategy

class Affine(Layer):
    def __init__(self, 
        *,
        inp,
        out,
        weight = None,
        bias = None,
        update_strategy = None,
        **kwargs
    ):
        self.inp = inp
        self.out = out
        self.weight = WeightInitializer.initialize(inp, out, weight)
        self.bias = BiasInitializer.initialize(out, bias)
        self.update_strategy = UpdateStrategy.create(update_strategy)
        self.last_shape = None
        self.inp_reinited = False
    def reshape(self, x):
        self.last_shape = x.shape
        s = x.size
        d = x.shape[0]
        return x.reshape(d, int(s / d))
    def restore_shape(self, x):
        if self.last_shape is None:
            return x
        shape = self.last_shape
        self.last_shape = None
        return x.reshape(shape)
    def fp(self, x):
        if x.ndim > 2:
            x = self.reshape(x)
        self.last_inp = x
        return self.predict(x)
    def predict(self, x):
        if x[0].size != self.inp:
            if self.inp_reinited:
                raise Exception('size of input for affine layer changed more than once')
            self.inp_reinited = True
            self.inp = x.shape[0].size
            self.weight = WeightInitializer.initialize(self.inp, self.out)
        return x @ self.weight + self.bias
    def bp(self, prp, *args, **kwargs):
        modification = self.last_inp.T @ prp
        modification += (kwargs.get('weight_decay') or 0) * self.weight
        self.update_strategy.calc(modification)
        return self.restore_shape(prp @ self.weight.T)
    def weight_sum(self):
        return np.sum(self.weight)
    def update(self):
        # print(self.weight)
        diff = self.update_strategy.update()
        self.weight = self.weight + diff
        return diff
