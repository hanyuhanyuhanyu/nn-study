from .layer import Layer
from .activator import Activator
from .initialization_strategy import create_weight, create_bias
from .update_strategy import UpdateStrategy

class Affine(Layer):
    @classmethod
    def createAffine(cls, setting):
        inp = setting.inp
        out = setting.out
        learn_rate = setting.learn_rate
        weight = create_weight(inp, out, setting.weight)
        bias = create_bias(inp, out, setting.bias)
        update_strategy = UpdateStrategy.create(setting)
        return Affine(
            inp,
            out,
            weight,
            bias,
            update_strategy
        )

    # 正しく初期値が与えられると信じる
    def __init__(self,
            inp,
            out,
            weight,
            bias,
            update_strategy,
            *_,
            **__,
        ):
            self.inp = inp
            self.out = out
            self.weight = weight
            self.bias = bias
            self.update_strategy = update_strategy
            self.prepareForBatch()

    # バッチを読む前にすべき処理
    # キャッシュの削除とか
    def prepareForBatch():        
        self.fp_stack = []
        self.bp_stack = []
    def fp(self, x, *_, **__):
        self.fp_stack.append(x)
        return self.predict(x, *_, **__)
    def predict(self, x, *_, **__):
        return x @ self.weight + self.bias
    def bp(self, prp, *args, **kwargs):
        self.bp_stack.append(prp)
        return prp @ self.weight.T
    def update(self, *args, **kwargs):
        diff = self.update_strategy(self) 
        self.weight += diff
        return diff
