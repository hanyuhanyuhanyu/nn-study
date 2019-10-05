import numpy as np

class UpdateStrategy:
    @classmethod
    def create(cls, setting):
        name = setting.get('name')
        kwargs = setting.get('setting') or {}
        if name == 'momentum':
            return Momentum(**kwargs)
        if name in ["rmsprop", "rms"]:
          return RMSProp(**kwargs)
        if name == 'adagrad':
          return Adagrad(**kwargs)
        if name == 'adadelta':
          return AdaDelta(**kwargs)
        if name == 'adam':
          return Adam(**kwargs)
        # if name in ['nesterov_accelerated_gradient', 'nesterov_ag', 'nag']:
        #     return NesterovAcceleratedGradient()
        return UpdateStrategy(**kwargs)
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

    def __init__(self, 
            *,
            learn_rate = 0.0125,
            **__
        ):
        self.learn_rate = learn_rate
        self.initialize()
    def initialize(self):
        self.learn_stack = []
    def calc(self, diff):
        self.learn_stack.append(diff)
    def update(self):
        upd = -self.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        self.initialize()
        return upd

class Momentum(UpdateStrategy):
    #rateはモーメンタム係数momentum coefficientのことだが名前が長すぎるので
    def __init__(self, *, learn_rate = 0.0125, rate = None, **kwargs):
        self.initialize()
        self.last_moment = None
        self.learn_rate = learn_rate
        self.rate = rate or 0.05
    def update(self):
        upd = -self.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        moment = self.last_moment if self.last_moment is not None else 0
        self.last_moment = len(self.learn_stack) * self.rate * moment + upd
        self.initialize()
        return self.last_moment

class RMSProp(UpdateStrategy):
    def __init__(self,
            *,
            learn_rate = 0.0125,
            epsilon = 0.1,
            attenuation_rate = 0.9, #減衰率
            **kwargs,
        ):
        self.initialize()
        self.last_ada = 0
        self.learn_rate = learn_rate
        self.attn = attenuation_rate or 0.9
        self.epsilon = epsilon or 0.1 #どれぐらいがいいのかさっぱりわからん
    def initialize(self):
        self.learn_stack = []
        self.ada_stack = []
    def calc_ada(self, diff):
        return self.attn * self.last_ada + (1 - self.attn) * diff ** 2
    def calc(self, diff):
        ada = self.calc_ada(diff)
        self.ada_stack.append(ada)
        self.learn_stack.append(diff / np.sqrt(self.epsilon + ada))
    def update(self):
        self.last_ada = np.sum(self.ada_stack, axis = 0)
        momentum = -self.learn_rate * np.sum(np.array(self.learn_stack), axis = 0)
        self.initialize()
        return momentum

class Adagrad(RMSProp):
    def calc_ada(self,  diff):
        return self.last_ada + diff * diff
    def calc(self, diff):
        ada = self.calc_ada(diff)
        self.ada_stack.append(ada)
        self.learn_stack.append(diff / (self.epsilon + np.sqrt(ada)))

#初期学習率がいらない
class AdaDelta(RMSProp):
    def __init__(self,
            **kwargs,
        ):
        super(AdaDelta, self).__init__(**kwargs)
        self.initialize()
        self.diff_mean = 0
    def initialize(self):
        super(AdaDelta, self).initialize()
        self.diff_stack = []
    def calc(self, diff):
        ada = self.calc_ada(diff)
        update_diff = -diff * np.sqrt(self.epsilon + self.diff_mean) / np.sqrt(self.epsilon + ada)
        self.ada_stack.append(ada)
        self.learn_stack.append(update_diff)
    def update(self):
        self.last_ada = np.sum(self.ada_stack, axis = 0)
        self.diff_mean = self.attn * self.diff_mean + (1 - self.attn) * np.sum(np.array(self.learn_stack) ** 2, axis = 0)
        momentum = np.sum(np.array(self.learn_stack), axis = 0)
        self.initialize()
        return momentum
class Adam(RMSProp):
    def __init__(self,
            *,
            learn_rate = 0.0125,
            epsilon = 0.1,
            attenuation_rate = 0.9, #減衰率
            **kwargs,
        ):
        self.initialize()
        self.learn_rate = learn_rate
        self.moment_first = 0
        self.moment_second = 0
        self.attn = attenuation_rate or 0.9
        self.attn_multipled = self.attn
        self.epsilon = epsilon or 0.1 #どれぐらいがいいのかさっぱりわからん
    def initialize(self):
        self.moment_1_stack = []
        self.moment_2_stack = []
    def calc(self, diff):
        mom_1 = self.attn * self.moment_first + (1 - self.attn) * diff
        mom_2 = self.attn * self.moment_second + (1 - self.attn) * diff * diff
        self.moment_1_stack.append(mom_1)
        self.moment_2_stack.append(mom_2)
    def update(self):
        self.moment_first = np.sum(np.array(self.moment_1_stack),axis=0)
        self.moment_second = np.sum(np.array(self.moment_2_stack),axis=0)
        molec = self.moment_first / (1 - self.attn_multipled)
        denomi = self.moment_second / (1 - self.attn_multipled)
        momentum = -self.learn_rate * molec / denomi
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
