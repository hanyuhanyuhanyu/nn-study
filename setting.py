from .mini_batch_strategy import MiniBatchStrategy
from .layer_factory import LayerFactory
from .loss_function import LossFunction
from .descriptor import Descriptor
class Setting:
    @classmethod
    def forTest(cls):
        inp = [
            [.1, .2, .3,],
            [.5, .0, .4,],
        ]
        out = [
            [0, -.2,],
            [.7, .1,],
        ]
        layers_setting = {
            'layers': [
                {
                    'setting': {
                        'affine': {
                            'inp': 3,
                            'out': 4,
                            'weight': 'xavier',
                            'update_strategy': {
                                'name': 'momentum',
                                'setting': {
                                    'learn_rate': 0.1,
                                },
                            },
                        },
                        'activator': {
                            'func': 'tanh',
                        },
                        'batch_regulator': {
                            'inp': 4,
                        },
                    },
                },
                {
                    'setting': {
                        'affine': {
                            'inp': 4,
                            'out': 4,
                            'weight': 'xavier',
                            'update_strategy': {
                                'name': 'momentum',
                                'setting': {
                                    'learn_rate': 0.1,
                                    'epsilon': 0.2,
                                    'attenuation_rate': 0.91,
                                },
                            },
                        },
                        'activator': {
                            'func': 'tanh',
                        },
                        'batch_regulator': {
                            'inp': 4,
                        },
                    },
                },
                {
                    'setting': {
                        'affine': {
                            'inp': 4,
                            'out': 2,
                            'weight': 'xavier',
                            'update_strategy': {
                                'name': 'momentum',
                                'setting': {
                                    'learn_rate': 0.1,
                                    'rate': 0.25,
                                },
                            },
                        },
                        'activator': {
                            'func': 'tanh',
                        },
                        'batch_regulator': {
                            'inp': 2,
                        },
                    },
                },
            ]
        }
        mini_batch_strategy_setting = {
            'epoch': 300
        }
        return Setting(inp, out,
            layers_setting = layers_setting,
            mini_batch_strategy_setting = mini_batch_strategy_setting,
        )

    def __init__(self, inp, out, **kwargs):
        self.inp = inp
        self.out = out
        # 読んだ時にNoneになって欲しいものを設定しているだけ
        # こいつらは基本public
        self.mini_batch_strategy_setting = kwargs.get('mini_batch_strategy_setting')
        self.layers_setting = kwargs.get('layers_setting')
        self.layer_default = None
        self.loss_setting = None
        self.descriptor_setting = None
    def create_mini_batch(self):
        return MiniBatchStrategy.create(self.inp, self.out, self.mini_batch_strategy_setting)
    def create_layers(self):
        return LayerFactory.create(self.layers_setting, self.layer_default)
    def create_loss_function(self):
        return LossFunction.create(self.loss_setting)
    def create_descriptor(self):
        return Descriptor.create(self.descriptor_setting)
