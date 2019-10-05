from .setting import Setting
import time
import numpy as np
from .batch_regulator import BatchRegulator
class LearningMachine:
    def __init__(self, setting):
        self.setting = setting
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
                descriptor.add_loss_history(loss)
                bp = loss_func.bp(fp, data.out)
                layers.bp(bp)
                layers.update()
        descriptor.descript(self)

def test():
    LearningMachine(Setting.forTest()).learn()

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
