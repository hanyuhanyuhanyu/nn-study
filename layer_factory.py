from .layer import Layer, AggregatedLayer, create_learning_layer
from .affine import Affine
from .activator import Activator

class LayerFactory:
    @classmethod
    def create(cls, settings = {}, default = None):
        default = default or {}
        layers = []
        for l in settings.get('layers') or []:
            name = l.get('name')
            setting = dict(default, **(l['setting'] or {}))
            layers.append(dispatch_layer_creator(name)(**setting))
        return AggregatedLayer(layers = layers)

def dispatch_layer_creator(name):
    if name is None:
        return lambda **setting: LayerFactory.create(create_learning_layer(**setting))
    name = name.lower()
    if name == 'affine':
        return Affine
    if name == 'activator':
        return Activator.create
