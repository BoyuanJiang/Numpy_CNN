# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/30/18

import numpy as np

class ReLU(object):
    def forward(self, x):
        out = np.maximum(0, x)
        cache = x
        return out, cache

    def gradient(self, dout, cache):
        x = cache
        dx = np.array(dout)
        dx[x<=0] = 0
        return dx
