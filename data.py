import numpy as np
import random

class Data:
    def __init__(
            self,
            inp,
            out,
        ):
        inp = np.array(inp)
        out = np.array(out)
        if(inp.ndim == 1):
            self.inp = np.array([inp])
            self.out = np.array([out])
        else:
            self.inp = np.array(inp)
            self.out = np.array(out)
        if(self.inp.shape[0] != self.out.shape[0]):
            raise Exception('input size and out size does not match')
        self.in_size = self.inp.shape[1]
        self.out_size = self.out.shape[1]
    def shuffle(self):
        inds = np.random.permutation(self.inp.shape[0])
        self.inp = self.inp[inds]
        self.out = self.out[inds]
    def sample(self, num = 5):
        inds = np.random.choice(self.inp.shape[0], num, False)
        return {'inp': self.inp[inds], 'out': self.out[inds]}
