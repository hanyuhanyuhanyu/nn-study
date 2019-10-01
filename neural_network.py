import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss
from .descriptor import Descriptor

class NN:
    def __init__(
            self,
            in_size,
            *,
            loss = 'square',
            out_func = 'id',
            func = 'tanh',
            learn_rate = 0.1,
            update_strategy = None
        ):
        self.in_size = in_size
        self.last_out_size = in_size
        self.out_func_name = out_func
        self.out_func_layer = None
        self.learn_rate = learn_rate
        self.func = func
        self.update_strategy = update_strategy
        self.layers = [
            Layer.create(
                'id',
                self.in_size,
                self.in_size,
            )
        ]
        self.loss_func = Loss.create(loss)
    def add_layer(self,
            layer,
            out_size,
            *,
            func = None,
            weight = None,
            bias = None,
            learn_rate = None,
            update_strategy = None,
            **kwargs):
        last_out_size = self.last_out_size
        self.last_out_size = out_size
        self.layers.append(
            Layer.create(
                layer,
                last_out_size,
                out_size,
                weight = weight,
                bias = bias,
                learn_rate = learn_rate or self.learn_rate or None,
                update_strategy = update_strategy or self.update_strategy,
                **kwargs,
            )
        )
        self.layers.append(
            Activator.create(
                func or self.func,
                **kwargs,
            )
        )
    def out_func(self):
        if(self.out_func_layer is None):
            self.out_func_layer = Activator.create(self.out_func_name)
        return self.out_func_layer
    def fp(self, inp):
        inp = np.array(inp)
        for layer in self.layers:
            inp = layer.fp(inp)
        inp = self.out_func().fp(inp)
        return inp
    def bp(self, predicted, expected):
        propagated = self.loss_func.bp(predicted, np.array(expected))
        propagated = self.out_func().bp(propagated)
        for i in range(len(self.layers)):
            ans = self.layers[-1 - i].bp(propagated)
            propagated = ans
    def update(self):
        for i in self.layers:
            i.update()
    def loss(self, predicted, expected):
        loss = self.loss_func.fp(predicted, np.array(expected))
        return np.sum(loss) / loss.size

