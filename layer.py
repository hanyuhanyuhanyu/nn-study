class Layer:
    def __init__(self, **kwargs):
        self.initial_setting = kwargs
    def copy():
        return Layer(**self.initial_setting)
    def fp(self, inp):
        return self.predict(inp)
    def predict(self, inp):
        return inp
    def bp(self, prop):
        return prop
    def update(self):
        pass

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
    def bp(self, prop):
        for l in reversed(self.layers):
            prop = l.bp(prop)
        return prop
    def update(self):
        for l in self.layers:
            l.update()


def create_learning_layer(
    *,
    affine = None,
    batch_regularization = False,
    activator = None,
):
    settings = [
        {
            'name': 'affine',
            'setting': affine
        },
    ]
    # todo
    # if batch_regularization:
    #     settings.append()
    settings.append({
        'name': 'activator',
        'setting': activator
    })
    return {'layers': settings}
