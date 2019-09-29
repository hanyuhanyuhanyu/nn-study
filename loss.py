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
    @classmethod
    def funcs(_):
        return[
            'id',
            'cross',
            'square'
        ]

class Id:
    def __init__(self, *args, **kargs):
        pass
    def fp(self, predicted, expected): #forward propagation
        return predicted
    def bp(self, predicted, expected): #forward propagation
        return predicted
    def update(self):
        pass
    def num_diff(self, pre, exp):
        return error_num_diff(self.fp, pre, exp)

class Cross(Id):
    def fp(self, pre, exp):
        axs = pre.ndim - 1
        return -np.sum(exp * np.log(np.clip(pre, 1e-7, None)), axis = axs)
    def bp(self, pre, exp):
        return -exp / np.clip(pre, 1e-7, None)

class Square(Id):
    def fp(self, pre, exp):
        axs = pre.ndim - 1
        return np.sum((pre - exp) ** 2, axis = axs) / 2
    def bp(self, pre, exp):
        return pre - exp
