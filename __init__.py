import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss

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
    def add_layer(self, layer, out_size, *, func = None, weight = None, bias = None):
        last_out_size = self.last_out_size
        self.last_out_size = out_size
        self.layers.append(
            Layer.create(
                layer,
                last_out_size,
                out_size,
                weight = weight,
                bias = bias,
                learn_rate = self.learn_rate or None
            )
        )
        self.layers.append(
            Activator.create(
                func or self.func,
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
        return self.loss_func.fp(predicted, np.array(expected))

def softmaxtest():
    def test(inp):
        axs = inp.ndim - 1
        mx = np.max(inp, axis=axs)
        print(inp)
        print(mx)
        di = np.exp(inp.T - mx)
        print(di)
        s = np.sum(di, axis=0)
        print(s)
        print((di / s).T)
    inp = np.array([
        [.1,.2,.3],
        [.5,.3,.2],
    ])
    test(inp)
    inp = np.array([.1,.2,.3])
    test(inp)
    
def testacts():
    prp = np.array([
        [1.,2.,3.],
        [2.5,1.5,0.5],
        [.5,.2,.3],
    ])
    inp = np.array([
        [.1,.2,.3],
        [.5,.3,.2],
        [-0.1,0,1.1],
    ])
    acts = Activator.funcs()
    print(inp)
    print(prp)
    for f in acts:
        print('testing', f)
        act = Activator.create(f)
        print(act.fp(inp))
        print(act.bp(1))
        print(act.bp(prp))

def testloss_tmp(predict, expect):
    funcs = Loss.funcs()
    print('test loss for')
    print('predict',predict)
    print('expect',expect)
    for f in funcs:
        l = Loss.create(f)
        print('test', f)
        print(l.fp(predict,expect))
        print(l.bp(predict,expect))

def testloss1():
    ps = [
        [0.2,.3,0.4],
        [1.,0.,0.],
    ]
    es = [
        [1.,0.,0.],
        [1.,0.,0.],
    ]
    for i in range(len(ps)):
        testloss_tmp(np.array(ps[i]), np.array(es[i]))

def testloss():
    predict = np.array(
        [
            [0.2,.3,0.4],
            [1.,0.,0.],
        ]
    )
    expect = np.array(
        [
            [1.,0.,0.],
            [1.,0.,0.],
        ]
    )
    testloss_tmp(predict,expect)

def affinetest():
    def test(mdl, inp, prp):
        print('inp')
        print(inp)
        print('prp')
        print(prp)
        print('---fp---')
        print(mdl.fp(inp))
        print('---bp---')
        print(mdl.bp(prp))
        print('---learned weight---')
        print(mdl.weight)
    inps = [
        [.1,.5,.8],
        [0,0,1],
        [
            [.1,.5,.8],
            [0,0,1],
        ],
    ]
    prps = [
        [.5, -0.5,],
        [1,-1,],
        [
            [.5, -0.5,],
            [1,-1,]
        ],
    ]
    weight = np.array([
        [1, 0.5,],
        [0.2, 0.3,],
        [0.3, 0.1,],
    ])
    bias = np.array([10, 0])
    print('---weight---')
    print(weight)
    print('---bias---')
    print(bias)
    print()
    print('///tests///')
    print()
    for i in range(len(inps)):
        inp = np.array(inps[i])
        prp = np.array(prps[i])
        size = inp.shape[inp.ndim - 1]
        l = Layer.create('affine', size, size, weight = weight, bias = bias)
        test(l, inp, prp)
        print()

def demo():
    n = NN(2, func = 'tanh', loss = 'cross', out_func='softmax', learn_rate = 0.5)
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
