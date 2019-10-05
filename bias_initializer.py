import numpy as np

class BiasInitializer:
    @classmethod
    def initialize(cls, out, kind = None):
        if kind in [None]:
            return np.random.rand(out) * 2. - 1.
        return kind
