import numpy as np
from .activator import Id

class WeightDitributionStrategy:
    @classmethod
    def create(cls, kind = None, *args, **kwargs):
        if(issubclass(kind.__class__, Uniform)):
            return kind
        if(issubclass(kind.__class__, Id)):
            kind = kind.category()
        if(kind in ['tanh', 'xavier']):
            return Xavier()
        elif(kind in ['relu', 'he']):
            return He()
        return Uniform()
class Uniform:
    def distribute(s, inp, out):
        print('uniform')
        return np.random.rand(inp,out)*2.-1.

class Xavier:
    def distribute(s, inp, out):
        print('xavier')
        return np.random.normal(0, np.sqrt(2 / (inp+out)), (inp, out))

class He:
    def distribute(s, inp, out):
        print('he')
        return np.random.normal(0, np.sqrt(2 / inp), (inp, out))
