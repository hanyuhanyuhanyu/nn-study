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
          return LeakyRelu(rate = kwargs['rate'])
        if func == 'softtanh':
          return SoftTanh(rate = kwargs['rate'])
        if func == 'softplus':
          return Softplus()
        if func == 'softmax':
          return SoftMax()
        if func == 'sigmoid':
            return Sigmoid()
        return Activator()

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
    def fp(self, x):
        self.last_inp = x
        print(x)
        print(self.rate)
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

