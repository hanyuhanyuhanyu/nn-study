import numpy as np
import matplotlib.pyplot as plt
class Descriptor:
    @classmethod
    def create(cls, kind = None, **kwargs):
        if(issubclass(kind.__class__, Id)):
            return kind
        if(kind == 'none'):
            return Id(**kwargs)
        elif(kind == 'sample'):
            return Sample(**kwargs)
        return Normal(**kwargs)
class Id:
    def __init__(self, *_, **__):
        pass
    def descript(*_, **__):
        return
class Normal(Id):
    def descript(self, nn_model, datas, answer_history, loss_history): 
        sample = datas.sample()
        inp = sample['inp']
        out = sample['out']
        print('inp')
        print(inp)
        print('out')
        print(out)
        print('last loss', loss_history[-1])
        x = range(len(loss_history))
        plt.figure()
        plt.plot(x, loss_history)
        plt.xlabel('iteration count')
        plt.ylabel('loss rate')
        plt.show()

class Sample(Id):
    def __init__(self, *, target_attr_name = 'default_func'):
        self.target_attr = target_attr_name
    def descript(self, nn_model, datas, answer_history, loss_history): 
        print(getattr(nn_model, self.target_attr)(), '/ last loss =>', loss_history[-1])
        x = range(len(loss_history))
        plt.figure()
        plt.plot(x, loss_history)
        plt.xlabel('iteration count')
        plt.ylabel('loss rate')
        plt.show()
