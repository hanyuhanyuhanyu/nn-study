import numpy as np
from .activator import Id

class WeightInitializer:
    @classmethod
    def distribute(cls, kind = None, *args, **kwargs):
        if(issubclass(kind.__class__, Uniform)):
            return kind.distribute()
        elif(type(kind) is str):
            cls.create(kind, *args, **kwarga).distribute()
        return kind or 1
    @classmethod
    def create(cls, kind = None, *args, **kwargs):
        if(kind in ['tanh', 'xavier']):
            return Xavier(*args, **kwargs)
        elif(kind in ['relu', 'he']):
            return He(*args, **kwargs)
        return Uniform(*args, **kwargs)

class Uniform:
    def distribute(s, inp, out):
        return np.random.rand(inp,out)*2.-1.

class Xavier:
    def distribute(s, inp, out):
        return np.random.normal(0, np.sqrt(2 / (inp+out)), (inp, out))

class He:
    def distribute(s, inp, out):
        return np.random.normal(0, np.sqrt(2 / inp), (inp, out))

