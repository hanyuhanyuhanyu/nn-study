"""
学習層の設定
layer_setting
    inp: number
    out: number
    activator: activator
    weight: string | weight(ただの行列)
    bias: string | bias(ただの行列)
    learn_rate: float
    update_strategy: update_strategy
    does_batch_regulalization: boolean

学習戦略の設定
inp: number
out: number
default_layer: layer_setting
layers: [layer_setting]
batch_strategy: batch_strategy
dwscriptor: descriptor
loss_setting: loss_setting
"""

# def create_layer():
#   数字
#     pass


class Setting:
    # 出力サイズはないと計算に失敗するので必須
    # 後は全部任意
    def __init__(self, **k):

        # 通したいレイヤーのセッティング
        self.layers = kwargs.get('layers')

        # 最後に通すレイヤー。以下２つのkeyのどちらかが与えられていることが必須
        last_layer = self.read_last_layer(**k)

        acceptable_settings = [
            'default_layer',
            'batch_strategy',
            'descriptor',
            'loss_setting',
        ]
        for setting in acceptable_settings:
            setattr(self, setting, kwargs.get(setting))
    def read_last_layer(self, **kwargs):
        # layerの最後 > default_layer > out_sizeの順に見る
        # 最低どれか一個はないと死ぬ
        if(self.layers is not None):
            return self.layers.pop(-1)
        return kwargs.get('default_layer') or k['out']
