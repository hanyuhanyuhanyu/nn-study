import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss
from .descriptor import Descriptor
from .neural_network import NN
from .data import Data
from .question_type import QuestionType
from .mini_batch_strategy import MiniBatchStrategy
from .test_data import * 

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
            descriptor = descriptor,
            mini_batch_strategy = 'flat',
        ):
        question_setting = QuestionType.setting(question)
        self.out_func = out_func or question_setting['out_func'] or 'id'
        self.loss = out_func or question_setting['loss'] or 'square'
        self.iteration = iteration
        #default values
        self.learn_rate = learn_rate
        self.descriptor = descriptor
        self.func = func
        self.weight = weight
        self.bias = bias
        #settings
        self.layer_settings = []
        self.mini_batch_strategy = MiniBatchStrategy.create(mini_batch_strategy)
        self.nn = None
        #history
        self.answer_history = []
        self.loss_history = []
    def default_func(self):
        if(type(self.func) is str):
            return self.func
        return self.func.__class__.__name__
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
        for i in range(self.iteration):
            epoch = self.mini_batch_strategy.make_epoch(self.data)
            for d in epoch:
                inp = d.inp
                out = d.out
                ans = self.nn.fp(inp)
                loss = self.nn.loss(ans, out)
                self.answer_history.append(ans)
                self.loss_history.append(loss)
                self.nn.bp(ans, out)
            self.nn.update()
        descriptor = Descriptor.create(descriptor or self.descriptor)
        descriptor.descript(
            self,
            self.data,
            self.answer_history,
            self.loss_history
        )
        return self.loss_history[-1]

def lm_sample():
    inp = [
        [.1,.2,.3],
        [.5,.0,.9],
        [.4,.1,.8],
    ]
    out = [
        [.2,.5],
        [.8,.3],
        [.4,.9],
    ]
    for func in Activator.list_up():
        lm = LeaningMachine(func = func, learn_rate = 0.2, descriptor = 'sample', mini_batch_strategy = None)
        lm.add_layer('affine', 3)
        lm.add_layer('affine', 2)
        lm.learn(inp, out)

def lm_test():
    d = test_data()
    inp = d["inp"]
    out = d["out"]
    mini_batch_strategy = MiniBatchStrategy.create('flat', blocks = 20)
    lm = LeaningMachine(func = 'sigmoid', learn_rate = 0.01, mini_batch_strategy = mini_batch_strategy)
    lm.add_layer('affine', 10)
    lm.add_layer('affine', 7)
    lm.add_layer('affine', 5)
    lm.add_layer('affine', 2)
    lm.learn(inp, out)
def lm_sin():
    d = test_sin()
    inp = d["inp"]
    out = d["out"]
    mini_batch_strategy = MiniBatchStrategy.create('flat', blocks = 20)
    lm = LeaningMachine(func = 'tanh', learn_rate = 0.05, mini_batch_strategy = mini_batch_strategy)
    lm.add_layer('affine', 10, func = 'sigmoid')
    lm.add_layer('affine', 7, func = 'relu')
    lm.add_layer('affine', 5)
    lm.add_layer('affine', 1)
    lm.learn(inp, out)
