# fw - (w % stride) % fw
# amari - size
import numpy as np
import math
def pad(imgs, h, w, fh, fw, stride):
    height_lack = math.ceil((h - fh) / stride) * stride + fh - h
    width_lack =  math.ceil((w - fw) / stride) * stride + fw- w

    pad_hu = math.floor(height_lack / 2)
    pad_hb = height_lack - pad_hu
    pad_wl = math.floor(width_lack / 2)
    pad_wr = width_lack - pad_wl
    return np.pad(imgs, ((0, 0), (0, 0), (pad_hu, pad_hb), (pad_wl, pad_wr)), 'constant')

def im2col(imgs, filter_height, filter_width, stride):
    data_count, channel_count, height, width = imgs.shape
    imgs = pad(imgs, height, width, filter_height, filter_width, stride)
    _, __, height, width = imgs.shape
    height_count = (math.floor((height - filter_height) / stride) + 1)
    width_count = (math.floor((width - filter_width) / stride) + 1)
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
    return ret

def col2im(cols, height, width, channel_count, filter_height, filter_width, stride):
    data_count, ch, cw = cols.shape
    ret = np.zeros((data_count, channel_count, height, width), float)
    i = 0
    s = filter_height * filter_width
    height_count = (math.floor((height - filter_height) / stride) + 1)
    width_count = (math.floor((width - filter_width) / stride) + 1)
    for h in range(filter_height):
        for w in range(filter_width):
            height_end = math.floor(height / stride) * stride + h
            width_end = math.floor(width / stride) * stride + w 
            this_get = cols[:, :, i::s].transpose(0,2,1).reshape(data_count, channel_count, width_count, height_count)
            ret[:, :, h:height_end:stride, w:width_end:stride] = this_get
            i += 1
    return ret


inp = np.array([
    [
        [
            [0,1,2,3,4],
            [5,6,7,8,9],
            [10,11,12,13,14],
            [15,16,17,18,19],
            [20,21,22,23,24],
        ],
        [
            [0,0,0,0,0],
            [1,1,1,1,1],
            [0,0,0,0,0],
            [1,1,1,1,1],
            [0,0,0,0,0],
        ],
    ],
    [
        [
            [.0,.1,.2,.3,.4],
            [.5,.6,.7,.8,.9],
            [1.0,1.1,1.2,1.3,1.4],
            [1.5,1.6,1.7,1.8,1.9],
            [2.0,2.1,2.2,2.3,2.4],
        ],
        [
            [1,2,1,2,1],
            [2,1,2,1,2],
            [1,2,1,2,1],
            [2,1,2,1,2],
            [1,2,1,2,1],
        ],
    ]
])

# print(inp)
col = im2col(inp,3,3,2)
print(col2im(col,5,5,2,3,3,2))
