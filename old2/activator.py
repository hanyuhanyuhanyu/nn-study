import numpy as np
import copy
from .layer import Layer
from .num_diff import num_diff

class Activator(Layer): 
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
            'softtanh',
        ]

    @classmethod
    def create(self, func, *args, **kwargs):
        #そもそも活性化関数が渡されているならそれを返す
        if(issubclass(func.__class__, Activator)):
            return copy.deepcopy(func)
        if func == 'tanh':
            return Tanh()
        elif func == 'hardtanh':
            return HardTanh()
        elif func == 'relu':
            return Relu()
        elif func == 'leaky_relu':
            return LeakyRelu(rate = kwargs['rate'])
        elif func == 'softtanh':
            return SoftTanh(rate = kwargs['rate'])
        elif func == 'softplus':
            return Softplus()
        elif func == 'softmax':
            return SoftMax()
        elif func == 'sigmoid':
            return Sigmoid()
        return Activator()

    @classmethod
    def list_up(cls):
        args = {}
        kwargs = {
            'leaky_relu': {
                'rate': 0.1
            },
            'softtanh': {
                'rate': 0.1
            }
        }
        funcs = []
        for f in cls.funcs():
            if(f in ['softmax']):
                continue
            arg = args.get(f) or []
            karg = kwargs.get(f) or {}
            funcs.append(cls.create(f, *args, **karg))
        return funcs

# tanh(x) 
# d/dx(tanh(x)) = 1 - tanh(x ** 2)
# のように、順伝播の結果を使い回せるようにすると計算が楽
# class softmax(Id):

class Softplus(Activator):
    def fp(self, x):
        self.last_calc = 1 + np.exp(x)
        self.last_inp = x
        return np.log(self.last_calc)
    def bp(self, prp):
        return prp / self.last_calc

class SoftMax(Activator):
    def fp(self, x):
        axs = x.ndim - 1
        mx = np.max(x, axis = axs)
        calc = np.exp(x.T - mx)
        self.last_result = (calc / np.sum(calc, axis=0)).T
        return self.last_result
    def bp(self, prp):
        return prp * self.last_result * (1 - self.last_result)

class Sigmoid(Activator):
    def fp(self, x):
        self.last_calc = np.exp(x * (-1))
        self.last_result = 1 / (1 + self.last_calc)
        return self.last_result
    def bp(self, prp):
        return prp * self.last_calc * (self.last_result ** 2)

class Tanh(Activator):
    def fp(self, x):
        self.last_inp = x
        self.last_result = np.tanh(x)
        return self.last_result
    def bp(self, prp):
        return prp * (1 - (self.last_result ** 2))
    def num_diff_func(self):
        return np.tanh

class HardTanh(Activator):
    def fp(self, x):
        self.last_inp = x
        return np.maximum(0, np.minimum(1, x))
    def bp(self, prp):
        x = self.last_inp
        return prp * (np.array((x > 0) * (x < 1)).astype(np.int))

class SoftTanh(Activator):
    def __init__(self, *, rate = 0):
        self.rate = rate
    def fp(self, x, *, rate = 0):
        self.last_inp = x
        return np.maximum(self.rate * x, np.minimum(self.rate * x, x))
    def bp(self, prp):
        x = self.last_inp
        x[(x > 0.) & (x < 1.)] = 1
        x[(x <= 0.) | (1 < x)] = self.rate
        return prp * x

class LeakyRelu(Activator):
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

class Relu(LeakyRelu):
    def __init__(self):
        super(LeakyRelu, self)
        self.rate = 0