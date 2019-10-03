import numpy as np
import copy

class UpdateStrategy:
    @classmethod
    def create(cls, kind, *_, **kwargs):
        if(issubclass(kind.__class__, Plain)):
            return copy.deepcopy(kind)
        if kind == 'momentum':
            return Momentum(**kwargs)
        elif kind == 'adagrad':
            return Adagrad(**kwargs)
        elif kind in ["rmsprop", "rms"]:
            return RMSProp(**kwargs)
        elif kind == 'adadelta':
            return AdaDelta(**kwargs)
        elif kind == 'adam':
            return AdaDelta(**kwargs)
        # elif kind in ['nesterov_accelerated_gradient', 'nesterov_ag', 'nag']:
        #     return NesterovAcceleratedGradient()
        return Plain()
    @classmethod
    def list_up(cls):
        return [
            'momentum',
            # 'nag',
            Adagrad(epsilon = 0.25),
            'rms',
            'adadelta',
            'adam',
        ]
class Plain:
    def __init__(self):
        self.learn_stack = []
    def initialize(self):
        self.learn_stack = []
    def calc(self, layer, prp):
        self.learn_stack.append(layer.last_inp.T @ prp)
    def update(self, layer):
        upd = -layer.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        self.initialize()
        return upd

class Momentum(Plain):
    #rateはモーメンタム係数momentum coefficientのことだが名前が長すぎるので
    def __init__(self, *, rate = None):
        self.initialize()
        self.last_moment = None
        self.rate = rate or 0.05
    def update(self, layer):
        upd = -layer.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        moment = self.last_moment if self.last_moment is not None else 0
        self.last_moment = len(self.learn_stack) * self.rate * moment + upd
        self.initialize()
        return self.last_moment

class RMSProp(Plain):
    def __init__(self,
            *,
            epsilon = None,
            attenuation_rate = None, #減衰率
        ):
        self.initialize()
        self.last_ada = 0
        self.attn = attenuation_rate or 0.9
        self.epsilon = epsilon or 0.1 #どれぐらいがいいのかさっぱりわからん
    def initialize(self):
        self.learn_stack = []
        self.ada_stack = []
    def calc_ada(self, layer, diff):
        return self.attn * self.last_ada + (1 - self.attn) * diff ** 2
    def calc(self, layer, prp):
        diff = layer.last_inp.T @ prp
        ada = self.calc_ada(layer, diff)
        self.ada_stack.append(ada)
        self.learn_stack.append(diff / np.sqrt(self.epsilon + ada))
    def update(self, layer):
        self.last_ada = np.sum(self.ada_stack, axis = 0)
        momentum = -layer.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        self.initialize()
        return momentum

class Adagrad(RMSProp):
    def calc_ada(self, layer, diff):
        return self.last_ada + diff * diff
    def calc(self, layer, prp):
        diff = layer.last_inp.T @ prp
        ada = self.calc_ada(layer, diff)
        self.ada_stack.append(ada)
        self.learn_stack.append(diff / (self.epsilon + np.sqrt(ada)))

#初期学習率がいらない
class AdaDelta(RMSProp):
    def __init__(self,
            *args,
            **kwargs,
        ):
        super(AdaDelta, self).__init__(*args, **kwargs)
        self.initialize()
        self.diff_mean = 0
    def initialize(self):
        super(AdaDelta, self).initialize()
        self.diff_stack = []
    def calc(self, layer, prp):
        diff = layer.last_inp.T @ prp
        ada = self.calc_ada(layer, diff)
        update_diff = -diff * np.sqrt(self.epsilon + self.diff_mean) / np.sqrt(self.epsilon + ada)
        self.ada_stack.append(ada)
        self.learn_stack.append(update_diff)
    def update(self, _):
        self.last_ada = np.sum(self.ada_stack, axis = 0)
        self.diff_mean = self.attn * self.diff_mean + (1 - self.attn) * np.sum(np.array(self.learn_stack) ** 2, axis = 0)
        momentum = np.sum(np.array(self.learn_stack), axis = 0)
        self.initialize()
        return momentum
class Adam(RMSProp):
    def __init__(self,
            *,
            epsilon = None,
            attenuation_rate = None, #減衰率
        ):
        self.initialize()
        self.moment_first = 0
        self.moment_second = 0
        self.attn = attenuation_rate or 0.9
        self.attn_multipled = self.attn
        self.epsilon = epsilon or 0.1 #どれぐらいがいいのかさっぱりわからん
    def initialize(self):
        self.moment_1_stack = []
        self.moment_2_stack = []
    def calc(self, layer, prp):
        diff = layer.last_inp.T @ prp
        mom_1 = self.attn * self.moment_first + (1 - self.attn) * diff
        mom_2 = self.attn * self.moment_second + (1 - self.attn) * diff * diff
        self.moment_1_stack.append(mom_1)
        self.moment_2_stack.append(mom_2)
    def update(self, layer):
        self.moment_first = np.sum(np.array(self.moment_1_stack),axis=0)
        self.moment_second = np.sum(np.array(self.moment_2_stack),axis=0)
        molec = self.moment_first / (1 - self.attn_multipled)
        denomi = self.moment_second / (1 - self.attn_multipled)
        momentum = -layer.learn_rate * molec / denomi
        self.initialize()
        self.attn_multipled *= self.attn
        return momentum

# 大文字シータを順伝播させた場合の勾配を求めるやり方がわからない
# class NesterovAcceleratedGradient(Plain):
#     #rateはモーメンタム係数momentum coefficientのことだが名前が長すぎるので
#     def __init__(self, *, rate = None):
#         self.learn_stack = []
#         self.last_moment = None
#         self.rate = rate or 0.05
#     def update(self, layer):
#         last_upd = self.last_upd 
#         upd = -layer.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
#         moment = self.last_moment if self.last_moment is not None else 0
#         alpha = len(self.learn_stack) * self.rate * moment
#         theta = layer.weight + self.last_moment
#         self.last_moment = self.rate * (self.last_moment + self.last_upd) + self.last_upd - self.last_moment
#         self.last_upd = upd
#         self.learn_stack = []
#         return self.last_moment
