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
        self.accuracy_history = []
    def add_loss_history(self, loss):
        self.loss_history.append(np.mean(loss))
    def add_accuracy(self, accuracy):
        self.accuracy_history.append(accuracy)
    def descript(self, model): 
        losses = self.loss_history
        print('last loss', losses[-1])
        print('last accuracy', self.accuracy_history[-1])
        x = range(len(losses))
        plt.figure()
        plt.plot(x, losses)
        plt.xlabel('iteration count')
        plt.ylabel('loss rate')
        plt.show()
        plt.figure()
        plt.plot(x, self.accuracy_history)
        plt.xlabel('iteration count')
        plt.ylabel('accuracy')
        plt.show()
