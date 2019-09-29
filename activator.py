import numpy as np
import copy
from .num_diff import num_diff

class Activator: 
    @classmethod
    def funcs(_):
        return [
            'id',
            'sigmoid',
            'relu',
            'leaky_relu',
            'softplus',
            'softmax',
            'tanh',
            'hardtanh',
        ]

    @classmethod
    def create(self, func, **kwargs):
        #そもそも活性化関数が渡されているならそれを返す
        if(issubclass(func.__class__, Id)):
            return copy.deepcopy(func)
        if func == 'tanh':
            return Tanh()
        elif func == 'hardtanh':
            return HardTanh()
        elif func == 'relu':
            return LeakyRelu()
        elif func == 'leaky_relu':
            return LeakyRelu(rate = kwargs['rate'])
        elif func == 'softplus':
            return Softplus()
        elif func == 'softmax':
            return SoftMax()
        elif func == 'sigmoid':
            return Sigmoid()
        return Id()

class Id:
    def __init__(self, *args, **kargs):
        self.last_result = None
        self.last_inp = None
    def fp(self, x): #forward propagation
        self.last_inp = x
        return x
    def bp(self, propagated): #back propagation
        return propagated
    def update(self):
        pass
    def num_diff_func(self):
        return self.fp
    def num_diff(self, inp):
        return num_diff(self.fp, self.last_inp)

# tanh(x) 
# d/dx(tanh(x)) = 1 - tanh(x ** 2)
# のように、順伝播の結果を使い回せるようにすると計算が楽
# class softmax(Id):

class Softplus(Id):
    def fp(self, x):
        self.last_calc = 1 + np.exp(x)
        self.last_inp = x
        return np.log(self.last_calc)
    def bp(self, prp):
        return prp / self.last_calc

class SoftMax(Id):
    def fp(self, x):
        axs = x.ndim - 1
        mx = np.max(x, axis = axs)
        calc = np.exp(x.T - mx)
        self.last_result = (calc / np.sum(calc, axis=0)).T
        return self.last_result
    def bp(self, prp):
        return prp * self.last_result * (1 - self.last_result)

class Sigmoid(Id):
    def fp(self, x):
        self.last_calc = np.exp(x * (-1))
        self.last_result = 1 / (1 + self.last_calc)
        return self.last_result
    def bp(self, prp):
        return prp * self.last_calc * (self.last_result ** 2)

class Tanh(Id):
    def fp(self, x):
        self.last_inp = x
        self.last_result = np.tanh(x)
        return self.last_result
    def bp(self, prp):
        return prp * (1 - (self.last_result ** 2))
    def num_diff_func(self):
        return np.tanh

class HardTanh(Id):
    def fp(self, x):
        self.last_inp = x
        return np.maximum(0, np.minimum(1, x))
    def bp(self, prp):
        x = self.last_inp
        return prp * (np.array((x > 0) * (x < 1)).astype(np.int))

class LeakyRelu(Id):
    def __init__(self, *, rate = 0): #rate = 0ならただのrelu
        self.rate = rate
    def fp(self, x):
        self.last_inp = x
        return np.maximum(self.rate * x, x)
    def bp(self, prp):
        x = self.last_inp
        x[x > 0.] = 1
        x[x <= 0.] = self.rate
        return prp * x

