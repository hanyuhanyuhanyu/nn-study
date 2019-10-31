from .layer import Layer, AggregatedLayer, create_learning_layer
from .affine import Affine
from .convolution import Convolution, Pooling
from .activator import Activator
from .batch_regulator import BatchRegulator

class LayerFactory:
    @classmethod
    def create(cls, settings = {}, default = None):
        default = default or {}
        layers = []
        give_to_next = {}
        for l in settings.get('layers') or []:
            name = l.get('name')
            setting = dict(default, **(l['setting'] or {}))
            setting.update(give_to_next)
            layer = dispatch_layer_creator(name)(**setting)
            setting = layer.give_to_next()
            layers.append(layer)
        return AggregatedLayer(layers = layers)

def dispatch_layer_creator(name):
    if name is None:
        return lambda **setting: LayerFactory.create(create_learning_layer(**setting))
    name = name.lower()
    if name == 'convolution':
        return Convolution
    if name == 'affine':
        return Affine
    if name == 'pooling':
        return Pooling.create
    if name == 'activator':
        return Activator.create
    if name == 'batch_regulator':
        return BatchRegulator
