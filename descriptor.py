import numpy as np
import matplotlib.pyplot as plt
class Descriptor:
    @classmethod
    def create(cls, kind = None):
        if(kind == 'none'):
            return Id()
        elif(kind == 'sample'):
            return Sample()
        return Normal()
class Id:
    def descript(*_, **__):
        return
class Normal:
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

class Sample:
    def descript(self, nn_model, datas, answer_history, loss_history): 
        print(nn_model.default_func())
        x = range(len(loss_history))
        plt.figure()
        plt.plot(x, loss_history)
        plt.xlabel('iteration count')
        plt.ylabel('loss rate')
        plt.show()
