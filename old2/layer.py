import numpy as np
from .strategy import (create_layer, NeuralNetworkStrategy)
from copy import deepcopy
from .num_diff import num_diff, h
from .update_strategy import UpdateStrategy
from .weight_distribution_strategy import WeightDitributionStrategy
from .weight_initializer import WeightInitializer

class Layer:
    def __init__(self, *args, **kwargs)
        pass
    def fp(self, inp, *args, **kwargs):
        self.predict(inp, *args, **kwargs)
    def bp(self, prp, *args, **kwargs):
        return prp
    def predict(self, inp, *args, **kwargs):
        return inp
    def update(self, update_strategy = None, *args, **kwargs):
        return 0
    def out_size(self):
        return None


class LayerArray(Layer):
    def __init__(self, strategy = None, *args, **kwargs):
        self.layers = []
        self.strategy = strategy or NeuralNetworkStrategy.create
    # Todo
    # def set_loss_layer():
    def add_layer(self, setting = None, *args, **kwargs):
        setting = setting or self.setting
        self.layers.append(create_layer(setting))
    def fp(self, inp, *args, **kwargs):
        self.last_prediction = self.predict(inp)
        return self.last_prediction
    def predict(self, inp, *args, **kwargs):
        prediction = inp
        for l in self.layers:
            prediction = l.fp(prediction)
        return prediction
    def bp(self, prp = None, *args, **kwargs):
        prp = prp or self.last_prediction
        for l in reversed(self.layers):
            prp = l.bp(prp)
            l.update()
        return prp

