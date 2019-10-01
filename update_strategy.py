import numpy as np

class UpdateStrategy:
    @classmethod
    def create(cls, kind, *_, **kwargs):
        if(issubclass(kind.__class__, Plain)):
            return kind
        if(kind == 'momentum'):
            return Momentum(**kwargs)
        return Plain()
class Plain:
    def __init__(self):
        self.learn_stack = []
    def calc(self, layer, prp):
        self.learn_stack.append(layer.last_inp.T @ prp)
        return 
    def update(self, layer):
        upd = -layer.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        self.learn_stack = []
        return upd
class Momentum(Plain):
    #rateはモーメンタム係数momentum coefficientのことだが名前が長すぎるので
    def __init__(self, *, rate = None):
        self.learn_stack = []
        self.last_moment = None
        self.rate = rate or 0.2
    def update(self, layer):
        upd = -layer.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        moment = self.last_moment if self.last_moment is not None else 0
        self.last_moment = len(self.learn_stack) * self.rate * moment + upd
        self.learn_stack = []
        return self.last_moment
