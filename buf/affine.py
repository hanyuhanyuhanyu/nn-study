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

    def fp(self, x):
        self.last_inp = x
        return self.predict(x)
    def predict(self, x):
        return x @ self.weight + self.bias
    def bp(self, prp, *args, **kwargs):
        modification = self.last_inp.T @ prp
        modification += (kwargs.get('weight_decay') or 0) * self.weight
        self.update_strategy.calc(modification)
        return prp @ self.weight.T
    def weight_sum(self):
        return np.sum(self.weight)
    def update(self):
        diff = self.update_strategy.update()
        self.weight = self.weight + diff
        return diff
