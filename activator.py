import numpy as np
from .num_diff import num_diff

class Activator: 
    @classmethod
    def create(self, func):
        if func == 'tanh':
            return Tanh()
        return Id()

class Id:
    def __init__(self, *args, **kargs):
        self.last_inp = None
    def fp(self, x): #forward propagation
        self.last_inp = x
        return x
    def bp(self, propagated): #back propagation
        return propagated
    def num_diff_func(self):
        return self.fp
    def num_diff(self, inp):
        return num_diff(self.fp, self.last_inp)

# tanh(x) 
# d/dx(tanh(x)) = 1 - tanh(x ** 2)
# のように、順伝播の結果を使い回せるようにすると計算が楽
class Tanh(Id):
    def __init__(self):
        self.last_result = None
        self.last_inp = None
        pass
    def fp(self, x):
        self.last_inp = x
        self.last_result = np.tanh(x)
        return self.last_result
    def bp(self, prp):
        return prp * (1 - (self.last_result ** 2))
    def num_diff_func(self):
        return np.tanh

class ActivationFunctions:
    @classmethod
    def sigmoid(cls, x, *_):
        return 1 / 1 + np.exp(-x)
    @classmethod
    def softmax(cls, x, *_):
        mx= np.max(x)
        exp = np.exp(x - mx)
        return exp / np.sum(exp)
    @classmethod
    def tanh(cls, x, *_):
        #tanh = (e ** x - e ** (-x)) / (e ** x + e ** (-1))
        return np.tanh(x)
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    @classmethod
    def relu(cls, x, *_):
        return np.maximum(0, x)
    # @classmethod
    # def leaky_relu(cls, x, *_):
    @classmethod
    def softplus(cls, x, *_):
        return np.log(1 + np.exp(x))
    @classmethod
    def hardtanh(cls, x, *_):
        return np.maximum(-1, np.minimum(1, x))
    @classmethod
    def id(cls, x, *_):
        return x
    @classmethod
    def square(cls, result, answer, *_):
        return np.sum((result - answer) ** 2) / 2
    @classmethod
    def cross(cls, result, answer, *_):
        return -np.sum(answer * np.log(np.clip(result, 1e-7, None)))

class Differential:
    @classmethod
    def sigmoid(cls, result, _ = None):
        sg = ActivationFunctions.sigmoid(result)
        return sg * (1 - sg)
    @classmethod
    def softmax(cls, result, _ = None):
        sm = ActivationFunctions.softmax(result)
        return sm * (1 - sm)
    @classmethod
    def tanh(cls, x, *_):
        return 1 - np.tanh(x) ** 2
    @classmethod
    def relu(cls, x, *_):
        return (x > 0).astype(np.int)
    # @classmethod
    # def leaky_relu(cls, x, *_):
    @classmethod
    def softplus(cls, x, *_):
        exp = np.exp(x)
        return exp / (1 + exp)
    @classmethod
    def hardtanh(cls, x, *_):
        return np.array((x > 0) * (x < 1)).astype(np.int)
    @classmethod
    def id(cls, x, *_):
        return 1
    @classmethod
    def cross(cls, result, answer):
        return -answer / result
    @classmethod
    def square(cls, result, answer):
        return result - answer

