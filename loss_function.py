import numpy as np
# cross entropy
class LossFunction:
    @classmethod
    def create(cls, setting = None):
        return LossFunction()
    def __init__(self):
        pass
    def fp(self, inp, out):
        return np.sum((inp - out) ** 2, axis = 1) / 2
    def bp(self, inp, out):
        return inp - out
