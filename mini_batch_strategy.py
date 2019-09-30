from .data import Data

class MiniBatchStrategy:
    @classmethod
    def create(cls, kind = None, **kwargs):
        if(kind == 'flat'):
            blocks = kwargs.get('blocks')
            return Flat(blocks = blocks)
        return MiniBatchStrategy()
    def __init__(self):
        pass

class Batch:
    def __init__(self, *_, **__):
        pass
    def make_epoch(self, data):
        return [Data(data.inp, data.out)]
class  Flat(Batch):
    def __init__(self, *, blocks = None):
        self.blocks = blocks or 5
    def make_epoch(self, data):
       data.shuffle() 
       inps = data.inp
       outs = data.out
       blocks = min(self.blocks, inps.shape[0])
       leng = inps.shape[0]
       datas = []
       for ind in range(blocks):
           i = inps[ind:leng:blocks]
           o = outs[ind:leng:blocks]
           datas.append(Data(i, o))
       return datas
