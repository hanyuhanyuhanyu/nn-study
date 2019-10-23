import numpy as np
from .layer import Layer
from .update_strategy import UpdateStrategy
from .bias_initializer import BiasInitializer

class BatchRegulator(Layer):
    def __init__(self,
        *, 
        inp,
        epsilon = 1e-8,
        update_strategy = None,
        **kwargs
    ):
        self.inp = inp
        # ema => Exponential Moving Average
        self.ema_rate = 0.875
        self.ema_of_mean = None
        self.ema_of_dispersion = None
        self.gamma = BiasInitializer.initialize(inp)
        self.beta = BiasInitializer.initialize(inp)
        self.gamma_update_strategy = UpdateStrategy.create(update_strategy or {})
        self.beta_update_strategy = UpdateStrategy.create(update_strategy or {})
        self.epsilon = epsilon
    def mean(self, inp):
        return np.mean(inp, axis = 0)
    def dispersion(self, inp):
        return np.std(inp, axis = 0) ** 2
    def calc_emas(self):
        if self.ema_of_mean is None:
            self.ema_of_mean = self.last_shifted
            self.ema_of_dispersion = self.last_dispersion
        else:
            self.ema_of_mean = self.ema_of_mean * self.ema_rate + self.last_mean * (1 - self.ema_rate)
            self.ema_of_dispersion = self.ema_of_dispersion * self.ema_rate + self.last_dispersion * (1 - self.ema_rate)
    def fp(self, inp):
        self.last_inp = inp
        self.last_mean = self.mean(inp)
        self.last_shifted = inp - self.last_mean
        self.last_dispersion = self.dispersion(inp)
        self.calc_emas()
        return self.predict(inp, self.last_shifted, self.last_dispersion)
    def predict(self, inp, shifted = None, dispersion = None):
        shifted = shifted if shifted is not None else inp - self.ema_of_mean
        dispersion = dispersion if dispersion is not None else self.ema_of_dispersion
        return self.beta + shifted * self.gamma / np.sqrt(dispersion + self.epsilon)
    def bp(self, prop, *_, **__):
        inp = self.last_inp
        n = inp.shape[1]
        shifted = self.last_shifted
        disp = self.last_dispersion
        eps = disp + self.epsilon
        sqrted = np.sqrt(eps)
        self.beta_update_strategy.calc(np.sum(prop, axis = 0))
        self.gamma_update_strategy.calc(np.sum(prop * shifted / sqrted, axis = 0))
        return (self.gamma / sqrted) * (1 - shifted ** 2 / (n * eps)) * (1 - 1 / n)
    def update(self):
        self.gamma += self.gamma_update_strategy.update()
        self.beta += self.beta_update_strategy.update()
