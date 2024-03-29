import numpy as np
import math
from .layer import Layer
from .weight_initializer import WeightInitializer
from .bias_initializer import BiasInitializer
from .update_strategy import UpdateStrategy

class Convolution(Layer):
    def __init__(self, 
        *,
        height,
        width,
        filter_height,
        filter_width,
        stride,
        channel_count,
        weight = None,
        bias = None,
        update_strategy = None,
        **kwargs
    ):
        self.height = height
        self.width = width
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.channel_count = channel_count
        self.weight = WeightInitializer.initialize(filter_height * filter_width * channel_count, channel_count, weight)
        self.bias = BiasInitializer.initialize(1, bias)[0]
        self.update_strategy = UpdateStrategy.create(update_strategy or {})
    def fp(self, x):
        return self.predict(x)
    def out_height(self):
        return (math.floor((self.height - self.filter_height) / self.stride) + 1)
    def out_width(self):
        return (math.floor((self.width - self.filter_width) / self.stride) + 1)
    def h_final(self, h):
        return self.stride + self.out_height() + h
    def w_final(self, w):
        return self.stride + self.out_width() + w
    def im2col(self, imgs):
        filter_height = self.filter_height
        filter_width = self.filter_width
        stride = self.stride
        data_count, channel_count, height, width = imgs.shape
        imgs = pad(imgs, height, width, filter_height, filter_width, stride)
        _, __, height, width = imgs.shape
        height_count = self.out_height()
        width_count = self.out_width()
        filter_size = filter_height * filter_width
        out_height = height_count * width_count
        out_width = channel_count * filter_size
        each_height = math.floor(height * width / (stride * stride))
        ret = np.zeros((data_count, out_height, out_width), float)
        i = 0
        for h in range(filter_height):
            for w in range(filter_width):
                this_time = imgs[:, :, h:h+height_count*stride:stride, w:w+width_count*stride:stride].reshape((data_count, channel_count, out_height)).transpose(0, 2, 1)
                ret[:, :, i::filter_size] += this_time
                i += 1
        self.last_col = ret
        return ret
    def predict(self, imgs):
        filter_height = self.filter_height
        filter_width = self.filter_width
        stride = self.stride
        data_count, channel_count, height, width = imgs.shape
        height_count = self.out_height()
        width_count = self.out_width()
        ret = self.im2col(imgs)
        return (ret @ self.weight).transpose(0,2,1).reshape(data_count, channel_count, height_count, width_count) + self.bias
    def col2im(self, cols):
        height = self.height
        width = self.width
        channel_count = self.channel_count
        filter_height = self.filter_height
        filter_width = self.filter_width
        stride = self.stride
        data_count, ch, cw = cols.shape
        ret = np.zeros((data_count, channel_count, height, width), float)
        i = 0
        s = filter_height * filter_width
        height_count = (math.floor((height - filter_height) / stride) + 1)
        width_count = (math.floor((width - filter_width) / stride) + 1)
        for h in range(filter_height):
            for w in range(filter_width):
                height_end = self.h_final(h)
                width_end = self.w_final(w)
                this_get = cols[:, :, i::s].transpose(0,2,1).reshape(data_count, channel_count, width_count, height_count)
                ret[:, :, h:h+height_count:stride, w:w+width_count:stride] += this_get
                i += 1
        return ret
    def bp(self, prp, **kwargs):
        d, c, h, w = prp.shape
        prp = prp.reshape(d, c, h*w).transpose(0,2,1)
        modification = self.last_col.transpose(0,2,1) @ prp
        modification += (kwargs.get('weight_decay') or 0) * self.weight
        self.update_strategy.calc(modification)
        return self.col2im(prp @ self.weight.T)

class Pooling(Convolution):
    def __init__(self, 
        *,
        height,
        width,
        filter_height,
        filter_width,
        stride,
        **kwargs
    ):
        self.height = height
        self.width = width
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
    def init_return_matrix(self, data, channel, out):
        self.count = 0
        self.ret = np.zeros((data, channel, out), float)
    def add_one_to_return_matrix(self, this_time):
        self.count += 1
        self.ret += this_time
    def finalize_return_matrix(self, imgs, d, c, h, w):
        self.back = np.zeros((d, c, self.height, self.width), float)
        return (self.ret / self.count).reshape(d,c,h,w)
    def fp(self, imgs):
        filter_height = self.filter_height
        filter_width = self.filter_width
        stride = self.stride
        data_count, channel_count, height, width = imgs.shape
        imgs = pad(imgs, height, width, filter_height, filter_width, stride)
        _, __, height, width = imgs.shape
        height_count = self.out_height()
        width_count = self.out_width()
        filter_size = filter_height * filter_width
        out_height = height_count * width_count
        out_width = channel_count * filter_size
        each_height = math.floor(height * width / (stride * stride))
        self.init_return_matrix(data_count, channel_count, out_height)
        i = 0
        for h in range(filter_height):
            for w in range(filter_width):
                this_time = imgs[:, :, h:h+height_count*stride:stride, w:w+width_count*stride:stride].reshape((data_count, channel_count, out_height))
                self.add_one_to_return_matrix(this_time)
                i += 1
        return self.finalize_return_matrix(imgs, data_count, channel_count, height_count, width_count)
    def bp(self, prp):
        d,c, _, __ = prp.shape
        prp /= self.count
        i = 0
        s = self.stride
        for h in range(self.filter_height):
            for w in range(self.filter_width):
                self.back[:, :, h:h+self.filter_height, w:w+self.filter_width] += prp[:, :, h, w].reshape(d, 1, 1, c)
                i += 1 
        return self.back
