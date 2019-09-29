import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss
from .descriptor import Descriptor

class NN:
    def __init__(
            self,
            in_size,
            *,
            loss = 'square',
            out_func = 'id',
            func = 'tanh',
            learn_rate = 0.1,
        ):
        self.in_size = in_size
        self.last_out_size = in_size
        self.out_func_name = out_func
        self.out_func_layer = None
        self.learn_rate = learn_rate
        self.func = func
        self.layers = [
            Layer.create(
                'id',
                self.in_size,
                self.in_size,
            )
        ]
        self.loss_func = Loss.create(loss)
    def add_layer(self, layer, out_size, *, func = None, weight = None, bias = None, learn_rate = None, **kwargs):
        last_out_size = self.last_out_size
        self.last_out_size = out_size
        self.layers.append(
            Layer.create(
                layer,
                last_out_size,
                out_size,
                weight = weight,
                bias = bias,
                learn_rate = learn_rate or self.learn_rate or None
            )
        )
        self.layers.append(
            Activator.create(
                func or self.func,
                **kwargs,
            )
        )
    def out_func(self):
        if(self.out_func_layer is None):
            self.out_func_layer = Activator.create(self.out_func_name)
        return self.out_func_layer
    def fp(self, inp):
        inp = np.array(inp)
        for layer in self.layers:
            inp = layer.fp(inp)
        inp = self.out_func().fp(inp)
        return inp
    def bp(self, predicted, expected):
        propagated = self.loss_func.bp(predicted, np.array(expected))
        propagated = self.out_func().bp(propagated)
        for i in range(len(self.layers)):
            ans = self.layers[-1 - i].bp(propagated)
            propagated = ans
    def loss(self, predicted, expected):
        loss = self.loss_func.fp(predicted, np.array(expected))
        return np.sum(loss) / loss.size


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

def demo():
    n = NN(2, func = 'tanh', loss = 'cross', out_func='softmax', learn_rate = 0.2)
    n.add_layer('affine', 2)
    n.add_layer('affine', 3)
    inp = [0.2, 0.5]
    test = [0.,0.,1.]
    for i in range(1000):
        print('---', i, '---')
        ans = n.fp(inp)
        print('predict', ans)
        print('loss', n.loss(ans, test))
        n.bp(ans, test)

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
