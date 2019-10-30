import numpy as np
import math
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
        self.weight_decay = None
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
        if self.use_batch_regulator_flag:
            self.layers.insert(0, 
                {
                    'name': 'batch_regulator',
                    'setting': {
                        'inp': self.inp,
                        'update_strategy': self.default_update_strategy
                    }
                }
            )
        self.created = Setting(
            inp = inp,
            out = out,
            layers_setting = {'layers': self.layers},
            mini_batch_strategy_setting = self.create_mini_batch_setting(),
            loss_setting = self.loss,
            weight_decay = self.weight_decay,
        )
        return self.created

#リスコフの置換原則バリバリに破ってるが知ったこっちゃねえ！ 
class SettingCreatorCnn(SettingCreator):
    def __init__(self,                                     
            height,
            width,
            channel,
            out_size,
            **kwargs,
        ):
        self.height = height
        self.width = width
        self.channel = channel
        self.out = out_size
        self.stride = kwargs.get('stride') or 2
        self.filter_channel = kwargs.get('filter_channel') or 5
        self.layers = []
        self.weight_decay = None
        self.closed = False
        self.default_filter_height = kwargs.get('filter_height') or 3
        self.default_filter_width = kwargs.get('filter_width') or 3
        self.set_default_update_strategy('momentum')
        self.use_batch_regulator_flag = True
        self.activator = 'tanh'
        self.loss = 'cross'
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
    def dont_use_batch_regulator(self):
        self.use_batch_regulator_flag =  False
    def create_layer_default_setting(self, **kwargs):
        fh = kwargs.get('filter_height') or self.default_filter_height
        fw = kwargs.get('filter_width') or self.default_filter_width
        s = kwargs.get('stride') or self.stride
        fc = kwargs.get('filter_channel') or self.filter_channel
        conv_setting = {
            'height': self.height,
            'width': self.width,
            'channel': self.channel,
            'filter_height': fh,
            'filter_width': fw,
            'stride': s,
            'filter_channel': fc,
            'weight': kwargs.get('weight') or Activator.initial_weight(self.activator),
            'update_strategy': self.default_update_strategy,
        }
        self.height = self.out_height(self.height, fh, s)
        self.width = self.out_width(self.width, fw, s)
        self.channel = fc
        pooling_setting = {
            'height': self.height,
            'width': self.width,
            'channel': fc,
            'filter_height': fh,
            'filter_width': fw,
            'stride': s,
        }
        self.height = self.out_height(self.height, fh, s)
        self.width = self.out_width(self.width, fw, s)
        return {
            'setting': {
                'convolution': conv_setting,
                'activator': {
                    'func': self.activator
                },
                'pooling': pooling_setting,
            }
        }
    def out_height(self, h, fh, s):
        return (math.floor((h - fh) / s) + 1)
    def out_width(self, w, fw, s):
        return (math.floor((w- fw) / s) + 1)
    def add_affine_layer(self, **kwargs):
        weight = kwargs.get('weight') or Activator.initial_weight(self.activator)
        self.layers.append({
            'setting': {
                'affine': {
                    'inp': self.height * self.width * self.channel,
                    'out': self.out,
                    'weight': weight,
                    'update_strategy': self.default_update_strategy,
                },
                'activator': {
                    'func': self.activator
                },
            }
        })
    def close(self, **kwargs):
        if self.closed:
            return
        kwargs = dict(kwargs, out = self.out)
        self.add_affine_layer(**kwargs)
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
        if(not self.closed):
            self.close()
        self.created = Setting(
            inp = inp,
            out = out,
            layers_setting = {'layers': self.layers},
            mini_batch_strategy_setting = self.create_mini_batch_setting(),
            loss_setting = self.loss,
            weight_decay = self.weight_decay,
        )
        return self.created
    
def modify_data_cnn(data):
    ret = []
    for d in data:
        ret.append((d < 96).astype(np.float))
    return np.array(ret)
def modify_data(data):
    ret = []
    for d in data:
        ret.append((d < 96).astype(np.int).flatten())
    return np.array(ret)
def modify_label(label):
    return label

class Setting:
    @classmethod
    def defaultTest(cls, inp, out):
        inp = np.array(inp)
        out = np.array(out)
        settings = SettingCreator(inp.shape[1], out.shape[1])
        settings.epoch_count = 200
        # b_r corrupted. below line required until that fixed
        settings.dont_use_batch_regulator()
        # settings.weight_decay = 0.2
        settings.set_default_update_strategy('rms', learn_rate=.01)
        settings.activator = 'relu'
        settings.add_layer(1, out = out.shape[1])
        settings.close()
        settings.loss = 'cross'
        return settings.create(inp, out)
    @classmethod
    def defaultTestCnn(cls, inp, out):
        inp = np.array(inp)
        out = np.array(out)
        _,channel,height,width = inp.shape
        out_size = out.shape[1]
        settings = SettingCreatorCnn(height, width, channel, out_size)
        settings.epoch_count = 1000
        # b_r corrupted. below line required until that fixed
        settings.dont_use_batch_regulator()
        # settings.weight_decay = 0.2
        settings.set_default_update_strategy('adagrad', learn_rate=.1)
        settings.activator = 'tanh'
        settings.add_layer(1, out = out.shape[1])
        settings.close()
        settings.loss = 'cross'
        return settings.create(inp, out)
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
        return Setting.defaultTest(inp, out)
    @classmethod
    def forTestCross(cls):
        inp = [
            [.1, .2, .3,],
            [-.5, .0, .4,],
            [.5, .1, .4,],
            [-.8, -.2, .0,],
        ]
        out = [
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [1., 0.],
        ]
        return Setting.defaultTest(inp, out)

    @classmethod
    def cnn(cls):
        dta = np.load('/workdir/nn/kadai/train_data.npy')
        lbl = np.load('/workdir/nn/kadai/train_label.npy')
        dta = modify_data_cnn(dta)
        lbl = modify_label(lbl)
        return Setting.defaultTestCnn(dta, lbl)
    @classmethod
    def kadai(cls):
        dta = np.load('/workdir/nn/kadai/train_data.npy')
        lbl = np.load('/workdir/nn/kadai/train_label.npy')
        dta = modify_data(dta)
        lbl = modify_label(lbl)
        return Setting.defaultTest(dta, lbl)

    def __init__(self, inp, out, **kwargs):
        self.inp = inp
        self.out = out
        # 読んだ時にNoneになって欲しいものを設定しているだけ
        # こいつらは基本public
        self.mini_batch_strategy_setting = kwargs.get('mini_batch_strategy_setting')
        self.layers_setting = kwargs.get('layers_setting')
        self.loss_setting = kwargs.get('loss_setting')
        self.layer_default = None
        self.weight_decay = kwargs.get('weight_decay')
        self.descriptor_setting = None
    def create_mini_batch(self):
        return MiniBatchStrategy.create(self.inp, self.out, self.mini_batch_strategy_setting)
    def create_layers(self):
        return LayerFactory.create(self.layers_setting, self.layer_default)
    def create_loss_function(self):
        return LossFunction.create(self.loss_setting)
    def create_descriptor(self):
        return Descriptor.create(self.descriptor_setting)
