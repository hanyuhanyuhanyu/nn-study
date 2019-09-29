import numpy as np
import matplotlib.pyplot as plt
class Descriptor:
    @classmethod
    def create(cls, kind = None):
        if(kind == 'none'):
            return Id()
        return Normal()
class Id:
    def descript(*_, **__):
        return
class Normal:
    def descript(self, nn_model, inp, out, answer_history, loss_history): 
        print('inp')
        print(inp)
        print('out')
        print(out)
        print('last answer')
        print(answer_history[-1])
        print('last loss', loss_history[-1])
        x = range(len(loss_history))
        plt.figure()
        plt.plot(x, loss_history)
        plt.xlabel('iteration count')
        plt.ylabel('loss rate')
        plt.show()
        for i in x:
            print('---', 1 + i ,'---')
            print(answer_history[i])
            print(loss_history[i])
