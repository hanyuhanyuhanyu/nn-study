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
        self.prepareForBatch()

    # バッチを読む前にすべき処理
    # キャッシュの削除とか
    def prepareForBatch(self):        
        self.last_inp = None
    def fp(self, x):
        self.last_inp = x
        return self.predict(x)
    def predict(self, x):
        return x @ self.weight + self.bias
    def bp(self, prp, *args, **kwargs):
        self.update_strategy.calc(self.last_inp, prp)
        return prp @ self.weight.T
    def update(self):
        diff = self.update_strategy.update()
        self.weight = self.weight + diff
        self.prepareForBatch()
        return diff
