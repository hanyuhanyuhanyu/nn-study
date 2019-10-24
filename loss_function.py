import numpy as np

# Square
class LossFunction:
    @classmethod
    def create(cls, setting = None):
        if setting in ['cross_entropy', 'cross']:
            return CrossEntropy()
        return LossFunction()
    def __init__(self):
        pass
    def fp(self, inp, out):
        return np.sum((inp - out) ** 2, axis = 1) / 2
    def bp(self, inp, out):
        return inp - out

class SoftMax:
    def fp(self, x):
        mx = np.max(x, axis = 1)
        calc = np.exp(x.T - mx)
        self.last_result = (calc / np.sum(calc, axis=0)).T
        return self.last_result
    def bp(self, prp, *_, **__):
        return prp * self.last_result * (1 - self.last_result)

class CrossEntropy(LossFunction):
    def __init__(self):
        self.softmax = SoftMax()
    def fp(self, inp, out):
        softmaxed = self.softmax.fp(inp)
        self.last_soft = softmaxed
        return (-np.sum(out * np.log(np.clip(softmaxed, 1e-12, None)), axis = 1)) / inp.shape[0]
    def bp(self, inp, out):
        return self.last_soft - out
        
def loss_test():
    arr = np.array([
        [10000,0,0],
        [0,10000,0],
    ])
    out = np.array(
        [
            [1,0,0],
            [0,1,0],
        ]
    )
    sft = SoftMax()
    print(sft.fp(arr))
    crs = CrossEntropy()
    print(crs.fp(arr, out))
    print(crs.bp(arr, out))

