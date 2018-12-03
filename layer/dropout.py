# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 12/2/18
import numpy as np

class Dropout(object):
    def __init__(self, keep_rate):
        self.keep_rate = keep_rate

    def forward(self, x, training=True):
        if training:
            mask = np.random.binomial(1, self.keep_rate, size=x.shape)/self.keep_rate
            mask_x = x*mask

            cache = mask
            return mask_x, cache
        else:
            return x

    def gradient(self, dout, cache):
        mask = cache
        dx = dout*mask
        return dx

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == '__main__':
    from utils.gradient_check import *
    dp = Dropout(0.8)

    inp = np.random.randn(2,3,32,32)
    out, cache = dp.forward(inp)
