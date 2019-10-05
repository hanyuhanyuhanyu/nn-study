import numpy as np

class WeightInitializer:
    @classmethod
    def initialize(cls, inp, out, kind = None):
        if kind == 'xavier':
            return np.random.normal(0, np.sqrt(2 / (inp+out)), (inp, out))
        elif kind == 'he':
            return np.random.normal(0, np.sqrt(2 / inp), (inp, out))
        elif kind in [None, 'uniform']:
            return  np.random.rand(inp, out) * 2. - 1.
        return kind

