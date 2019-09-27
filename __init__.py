import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss


class NN:
    def __init__(
            self,
            in_size,
            *,
            loss = 'square',
        ):
        self.in_size = in_size
        self.last_out_size = in_size
        self.layers = [
            Layer.create(
                'id',
                self.in_size,
                self.in_size,
            )
        ]
        self.loss_func = Loss.create(loss)
    def add_layer(self, layer, out_size, *, func = 'tanh', weight = None, bias = None):
        last_out_size = self.last_out_size
        self.last_out_size = out_size
        self.layers.append(
            Layer.create(
                layer,
                last_out_size,
                out_size,
                weight = weight,
                bias = bias,
            )
        )
        self.layers.append(
            Activator.create(
                func,
            )
        )
    def fp(self, inp):
        inp = np.array(inp)
        for layer in self.layers:
            inp = layer.fp(inp)
        return inp
    def bp(self, predicted, expected):
        propagated = self.loss_func.bp(predicted, np.array(expected))
        for i in range(len(self.layers)):
            ans = self.layers[-1 - i].bp(propagated)
            print(self.layers[-1 - i].__class__.__name__)
            print(ans)
            print(self.layers[-1 - i].num_diff(propagated))
            propagated = ans
    def loss(self, predicted, expected):
        return self.loss_func.fp(predicted, np.array(expected))
