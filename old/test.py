import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss

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

