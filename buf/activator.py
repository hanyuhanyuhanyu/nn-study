import numpy as np
from .layer import Layer
class Activator(Layer):
    @classmethod
    def create(cls, 
        *,
        func = None,
        **kwargs,
    ):
        if func == 'tanh':
            return Tanh()
        if func == 'hardtanh':
          return HardTanh()
        if func == 'relu':
          return Relu()
        if func == 'leaky_relu':
          return LeakyRelu(rate = kwargs.get('rate'))
        if func == 'softtanh':
          return SoftTanh(rate = kwargs.get('rate'))
        if func == 'softplus':
          return Softplus()
        if func == 'sigmoid':
            return Sigmoid()
        return Activator()
    @classmethod
    def initial_weight(cls, func_name):
        if func_name in ['tanh', 'sigmoid']:
            return 'xavier'
        if func_name in ['hardtanh', 'relu', 'leaky_relu', 'softtanh', 'softplus', 'sigmoid']:
            return 'he'
        return 'uniform'

class Softplus(Activator):
    def fp(self, x):
        self.last_calc = 1 + np.exp(x)
        self.last_inp = x
        return np.log(self.last_calc)
    def bp(self, prp, *_, **__):
        return prp / self.last_calc

class Sigmoid(Activator):
    def fp(self, x):
        self.last_calc = np.exp(x * (-1))
        self.last_result = 1 / (1 + self.last_calc)
        return self.last_result
    def bp(self, prp, *_, **__):
        return prp * self.last_calc * (self.last_result ** 2)

class Tanh(Activator):
    def fp(self, x):
        self.last_inp = x
        self.last_result = np.tanh(x)
        return self.last_result
    def bp(self, prp, *_, **__):
        return prp * (1 - (self.last_result ** 2))
    def num_diff_func(self):
        return np.tanh

class HardTanh(Activator):
    def fp(self, x):
        self.last_inp = x
        return np.maximum(0, np.minimum(1, x))
    def bp(self, prp, *_, **__):
        x = self.last_inp
        return prp * (np.array((x > 0) * (x < 1)).astype(np.int))

class SoftTanh(Activator):
    def __init__(self, *, rate = None):
        self.rate = rate or 0.1
    def fp(self, x):
        self.last_inp = x
        return np.maximum(self.rate * x, np.minimum(self.rate * x, x))
    def bp(self, prp, *_, **__):
        x = self.last_inp
        x[(x > 0.) & (x < 1.)] = 1
        x[(x <= 0.) | (1 < x)] = self.rate
        return prp * x

class LeakyRelu(Activator):
    def __init__(self, *, rate = None): #rate = 0ならただのrelu
        self.rate = rate or 0.1
    def fp(self, x):
        self.last_inp = x
        return np.maximum(self.rate * x, x)
    def bp(self, prp, *_, **__):
        x = self.last_inp
        x[x > 0.] = 1
        x[x <= 0.] = self.rate
        return prp * x

class Relu(LeakyRelu):
    def __init__(self):
        super(LeakyRelu, self)
        self.rate = 0

