import numpy as np
import matplotlib.pyplot as plt

class Descriptor:
    @classmethod
    def create(cls, setting = None):
        return Descriptor()
    def __init__(self):
        self.clear_history()
    def clear_history(self):
        self.loss_history = []
    def add_loss_history(self, loss):
        self.loss_history.append(np.mean(loss))
    def descript(self, model): 
        losses = self.loss_history
        print('last loss', losses[-1])
        x = range(len(losses))
        plt.figure()
        plt.plot(x, losses)
        plt.xlabel('iteration count')
        plt.ylabel('loss rate')
        plt.show()
