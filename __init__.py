import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss
from .descriptor import Descriptor
from .neural_network import NN

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

# classification => bunrui
# regression => caiki
class QuestionType:
    @classmethod
    def setting(cls, ques):
        if(ques in [ 'class', 'classification']):
            return {'out_func': 'softmax', 'loss': 'cross'}
        elif(ques in ['reg', 'regres', 'regression']):
            return {'out_func': 'id', 'loss': 'square'}
        return {'out_func': None, 'loss': None}

class LeaningMachine:
    def __init__(
            self,
            *,
            question = 'regression',
            out_func = None,
            loss = None,
            func = 'relu',
            learn_rate = 0.2,
            weight = None,
            bias = None,
            iteration = 1000,
        ):
        question_setting = QuestionType.setting(question)
        self.out_func = out_func or question_setting['out_func'] or 'id'
        self.loss = out_func or question_setting['loss'] or 'square'
        self.iteration = iteration
        #default values
        self.learn_rate = learn_rate
        self.func = func
        self.weight = weight
        self.bias = bias
        self.layer_settings = []
        self.answer_history = []
        self.loss_history = []
        self.nn = None
    def add_layer(self,
            *args,
            **kwargs,
        ):
        keys = [
            'learn_rate',
            'weight',
            'bias',
            'func',
        ]
        for k in keys:
            kwargs[k] = kwargs.get(k) or getattr(self, k, None)
        self.layer_settings.append({'args': args, 'kwargs': kwargs})
    def init_nn(self, inp, out, **kwargs):
        if(self.nn is not None):
            return
        self.data = Data(inp, out)
        kwargs['out_func'] = kwargs.get('out_func') or self.out_func
        kwargs['loss'] = kwargs.get('loss') or self.loss
        self.nn = NN(
            self.data.in_size,
            **kwargs
        )
        for setting in self.layer_settings:
            args = setting['args']
            kwargs = setting['kwargs']
            self.nn.add_layer(*args, **kwargs)

    def should_finish(self):
        return False
    def learn(
            self,
            inp,
            out,
            *,
            descriptor = None,
            **kwargs,
        ):
        self.init_nn(inp, out, **kwargs)
        inp = self.data.inp
        out = self.data.out
        for i in range(self.iteration):
            ans = self.nn.fp(inp)
            loss = self.nn.loss(ans, out)
            self.answer_history.append(ans)
            self.loss_history.append(loss)
            self.nn.bp(ans, out)
            self.nn.update()
            if(self.should_finish()):
                break
        if(descriptor is None):
            descriptor = Descriptor.create()
        descriptor.descript(
            self.nn,
            inp,
            out,
            self.answer_history,
            self.loss_history
        )
        return self.loss_history[-1]

def lm_test():
    inp = [
        [.2, .3,.4],
        [.5, .0, .1],
    ]
    out = [
        [.8,.4],
        [.1,.9],
    ]
    act = Activator.create('leaky_relu', rate = 0.2)
    lm = LeaningMachine(func = act, learn_rate = 0.1)
    lm.add_layer('affine', 3)
    lm.add_layer('affine', 2)
    lm.learn(inp, out)
