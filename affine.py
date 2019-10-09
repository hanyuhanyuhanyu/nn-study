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
        dropout_rate = None,
        **kwargs
    ):
        self.inp = inp
        self.out = out
        self.weight = WeightInitializer.initialize(inp, out, weight)
        self.bias = BiasInitializer.initialize(out, bias)
        self.update_strategy = UpdateStrategy.create(update_strategy)
        self.dropout_rate = dropout_rate or 0
    def create_dropout(self):
        return (np.array(self.out) >= self.dropout_rate).astype(np.int)
    def apply(self, inp):
        return inp @ self.weight + self.bias
    def fp(self, x):
        self.last_inp = x
        self.last_dropout = self.create_dropout()
        return self.apply(x) * self.last_dropout
    def predict(self, x):
        return self.apply(x) * (1 - self.dropout_rate)
    def bp(self, prp, *args, **kwargs):
        modification = self.last_inp.T @ prp
        modification += kwargs.get('weight_decay') or 0
        self.update_strategy.calc(modification, dropout = self.last_dropout)
        return prp @ self.weight.T * self.last_dropout
    def weight_sum(self):
        return np.sum(self.weight * np.array([self.last_dropout]).T)
    def update(self):
        diff = self.update_strategy.update()
        self.weight = self.weight + diff
        return diff
