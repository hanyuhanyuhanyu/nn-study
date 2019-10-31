from .setting import Setting
import time
import numpy as np
from .batch_regulator import BatchRegulator
from .loss_function import loss_test
def m():
    inp = np.array(
        [
            [
                [0,1,2],
                [1,2,0],
                [2,1,0],
            ],
            [
                [2,0,1],
                [1,0,2],
                [0,2,1],
            ]
        ]
    )
    hoge = np.zeros(inp.shape)
    d,c,_ = inp.shape
    print(inp.max(-1))
    mx = (inp.argmax(-1)[...,None] == np.arange(inp.shape[2])).astype(float)
    print(mx * inp)
class LearningMachine:
    def __init__(self, setting):
        self.setting = setting
        self.last_decay = None
    def learn(self):
        s = self.setting
        mini_batch = s.create_mini_batch()
        layers = s.create_layers()
        loss_func = s.create_loss_function()
        descriptor = s.create_descriptor()
        loss = None
        while mini_batch.should_continue():
            mini_batch.rewind()
            while mini_batch.remain():
                data = mini_batch.next()
                fp = layers.fp(data.inp)
                loss = loss_func.fp(fp, data.out)
                descriptor.add_accuracy(self.calc_accuracy(fp, data.out))
                descriptor.add_loss_history(loss)
                decay = self.weight_decay(layers)
                loss += decay 
                bp = loss_func.bp(fp, data.out)
                layers.bp(bp, weight_decay = self.weight_decay_for_bp(layers))
                layers.update()
        descriptor.descript(self)
    # 荷重減衰
    def weight_decay(self, layers):
        if self.setting.weight_decay is None:
            return 0
        self.last_decay = layers.weight_sum()
        return self.setting.weight_decay * self.last_decay ** 2 / 2
    def weight_decay_for_bp(self, layers):
        return self.setting.weight_decay or 0
    def calc_accuracy(self, inp, out):
        return np.mean((np.argmax(inp, axis = 1) == np.argmax(out, axis = 1)).astype(np.float))


def test():
    LearningMachine(Setting.forTest()).learn()
def kadai():
    LearningMachine(Setting.kadai()).learn()
def cross():
    loss_test()
def crossTest():
    LearningMachine(Setting.forTestCross()).learn()
def cnn(): 
    LearningMachine(Setting.cnn()).learn()

def exp():
    inp = [
        [.1, .2, 0.3,],
        [.5, .0, .4,],
    ]
    br = BatchRegulator(
        inp = 3,
        epsilon = 1e-8,
    )
    print(br.fp(np.array(inp)))
