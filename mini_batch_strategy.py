import numpy as np
from .data import Data

class MiniBatchStrategy:
    @classmethod
    def create(cls, inp, out, setting = None):
        setting = setting or {}
        # Todo: mini batchの戦略を切り替える
        return MiniBatchStrategy(inp, out, setting)
    def __init__(self, inp, out, setting = None):
        self.inp = np.array(inp)
        self.out = np.array(out)
        if self.inp.ndim == 1:
            self.inp = np.array([inp])
            self.out = np.array([out])
        self.epoch = setting.get('epoch') or 1000
        self.rewind()
    def rewind(self):
        self.iteration = 1
    def remain(self):
        return self.iteration > 0
    def next(self):
        ret = Data(self.inp, self.out)
        self.iteration -= 1
        return ret
    def should_continue(self):
        self.epoch -= 1
        return self.epoch > 0
