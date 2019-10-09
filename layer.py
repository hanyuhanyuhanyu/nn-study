class Layer:
    def __init__(self, **kwargs):
        self.initial_setting = kwargs
    def copy():
        return Layer(**self.initial_setting)
    def fp(self, inp):
        return self.predict(inp)
    def predict(self, inp):
        return inp
    def bp(self, prop, *args, **kwargs):
        return prop
    def update(self):
        pass
    def weight_sum(self):
        return 0

class AggregatedLayer(Layer):
    def __init__(self,
        *,
        layers = None
    ):
        self.layers = layers or [Layer()]
    def fp(self, inp):
        for l in self.layers:
            inp = l.fp(inp)
        return inp
    def predict(self, inp):
        for l in self.layers:
            inp = l.predict(inp)
        return inp
    def bp(self, prop, *args, **kwargs):
        for l in reversed(self.layers):
            prop = l.bp(prop, *args, **kwargs)
        return prop
    def update(self):
        for l in self.layers:
            l.update()
    def weight_sum(self):
        decay = 0
        for l in self.layers:
            decay += l.weight_sum()
        return decay

class LearningLayer(AggregatedLayer):
    def __init__(self,
        *,
        layers = None
    ):
        self.layers = layers or [Layer()]
        self.last_dropout = 1
    def fp(self, inp):
        affine = self.layers[0]
        inp = affine.fp(inp)
        self.last_dropout = affine.last_dropout
        for l in self.layers[1:]:
            inp = l.fp(inp, dropout = drp)
        return inp * drp
    def predict(self, inp):
        for l in self.layers:
            inp = l.predict(inp)
        return inp
    def bp(self, prop, *args, **kwargs):
        for l in reversed(self.layers):
            prop = l.bp(prop, *args, **kwargs)
        return prop
    def update(self):
        for l in self.layers:
            l.update()
    def weight_sum(self):
        decay = 0
        for l in self.layers:
            decay += l.weight_sum()
        return decay

def create_learning_layer(
    *,
    affine = None,
    batch_regulator = None,
    activator = None,
):
    settings = [
        {
            'name': 'affine',
            'setting': affine
        },
    ]
    if batch_regulator is not None:
        settings.append({
            'name': 'batch_regulator',
            'setting': batch_regulator,
        })
    settings.append({
        'name': 'activator',
        'setting': activator
    })
    return {'layers': settings}
