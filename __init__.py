from .setting import Setting
import time
import numpy as np
from .batch_regulator import BatchRegulator
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
                decay = self.weight_decay(layers)
                loss += decay 
                descriptor.add_loss_history(loss)
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
        if self.setting.weight_decay is None:
            return 0
        if self.last_decay is None:
            self.weight_decay(layers) # decayのキャッシュを裏でやっている
        decay = self.last_decay
        return self.setting.weight_decay * decay


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
