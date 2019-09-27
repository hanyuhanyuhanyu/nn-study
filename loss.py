import numpy as np
from .num_diff import error_num_diff

class Loss:
    @classmethod
    def create(self, func): 
        if func == 'square':
            return Square()
        elif func == 'cross':
            return Cross()
        return Id()

class Id:
    def __init__(self, *args, **kargs):
        pass
    def fp(self, predicted, expected): #forward propagation
        return predicted
    def bp(self, predicted, expected): #forward propagation
        return predicted
    def num_diff(self, pre, exp):
        return error_num_diff(self.fp, pre, exp)

class Cross(Id):
    def fp(self, pre, exp):
        return -np.sum(exp * np.log(np.clip(pre, 1e-7, None)))
    def bp(self, pre, exp):
        return -exp / pre

class Square(Id):
    def fp(self, pre, exp):
        return np.sum((pre - exp) ** 2) / 2
    def bp(self, pre, exp):
        return pre - exp
