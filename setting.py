from .activator import Activator
from .mini_batch_strategy import MiniBatchStrategy
from .layer_factory import LayerFactory
from .loss_function import LossFunction
from .descriptor import Descriptor

class SettingCreator:
    def __init__(self,
            inp_size,
            out_size,
            **kwargs,
        ):
        self.inp = inp_size
        self.out = out_size
        self.last_out = inp_size
        self.layers = []
        self.closed = False
        self.default_node_num = round((self.inp + self.out) * 0.75)
        self.set_default_update_strategy('momentum')
        self.use_batch_regulator_flag = True
        self.activator = 'tanh'
        self.loss = 'square'
        self.epoch_count = 100
        self.mini_batch = {
            'epoch': 100
        }
        self.created = None
    def add_layer(self,
            count = 1,
            **kwargs,
        ):
        for i in range(count):
            self.layers.append(self.create_layer_default_setting(**kwargs))
            self.last_out = kwargs.get('out') or self.default_node_num
    def dont_use_batch_regulator(self):
        self.use_batch_regulator_flag =  False
    def create_layer_default_setting(self, **kwargs):
        weight = kwargs.get('weight') or Activator.initial_weight(self.activator)
        out = kwargs.get('out') or self.default_node_num
        batch_regulator = {
            'inp': out,
        }
        if self.use_batch_regulator_flag is False:
            batch_regulator = None
        return {
            'setting': {
                'affine': {
                    'inp': self.last_out,
                    'out': out,
                    'weight': weight,
                    'update_strategy': self.default_update_strategy,
                },
                'activator': {
                    'func': self.activator
                },
                'batch_regulator': batch_regulator
            }
        }
    def close(self, **kwargs):
        if self.closed:
            return
        kwargs = dict(kwargs, out = self.out)
        self.add_layer(**kwargs)
        self.closed = True
    def set_default_update_strategy(self, name, **kwargs):
        self.default_update_strategy = self.create_default_update_strategy(name, **kwargs)
    def create_default_update_strategy(self, name, **kwargs):
        return {
            'name': name,
            'setting': kwargs,
        }
    def create_mini_batch_setting(self):
        return {
            'epoch': self.epoch_count,
        }
    def create(self, inp, out):
        if self.created is not None:
            return self.created
        if(self.last_out != self.out):
            self.close()
        # if self.use_batch_regulator_flag:
        #     self.layers.insert(0, 
        #         {
        #             'name': 'batch_regulator',
        #             'setting': {
        #                 'inp': self.inp,
        #                 'update_strategy': self.default_update_strategy
        #             }
        #         }
        #     )
        self.created = Setting(
            inp = inp,
            out = out,
            layers_setting = {'layers': self.layers},
            mini_batch_strategy_setting = self.create_mini_batch_setting(),
            loss_setting = self.loss,
        )
        return self.created
    
class Setting:
    @classmethod
    def forTest(cls):
        inp = [
            [.1, .2, .3,],
            [-.5, .0, .4,],
        ]
        out = [
            [.5, .2,],
            [-.3, 0.1,],
        ]
        settings = SettingCreator(3, 2)
        settings.epoch_count = 300
        # settings.dont_use_batch_regulator()
        settings.set_default_update_strategy('rms')
        settings.activator = 'tanh'
        settings.add_layer(1)
        settings.close()
        loss_setting = None
        return settings.create(inp, out)

    def __init__(self, inp, out, **kwargs):
        self.inp = inp
        self.out = out
        # 読んだ時にNoneになって欲しいものを設定しているだけ
        # こいつらは基本public
        self.mini_batch_strategy_setting = kwargs.get('mini_batch_strategy_setting')
        self.layers_setting = kwargs.get('layers_setting')
        self.loss_setting = kwargs.get('loss_setting')
        self.layer_default = None
        self.descriptor_setting = None
    def create_mini_batch(self):
        return MiniBatchStrategy.create(self.inp, self.out, self.mini_batch_strategy_setting)
    def create_layers(self):
        return LayerFactory.create(self.layers_setting, self.layer_default)
    def create_loss_function(self):
        return LossFunction.create(self.loss_setting)
    def create_descriptor(self):
        return Descriptor.create(self.descriptor_setting)
