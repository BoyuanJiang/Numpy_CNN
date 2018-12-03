# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/30/18

import numpy as np

class Linear(object):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = np.random.standard_normal((in_channels, out_channels))/np.sqrt(in_channels/2)
        self.b = np.zeros((out_channels,))
        self.gradient_W = np.zeros_like(self.W)
        self.gradient_b = np.zeros_like(self.b)
        self.weights_m = np.zeros_like(self.W)
        self.bias_m = np.zeros_like(self.b)

    def forward(self, x, w=None, b=None):
        if w is not None:
            self.W = w
        if b is not None:
            self.b = b
        assert len(x.shape)==2
        out = np.dot(x, self.W)+self.b
        return out, x

    def gradient(self, dout, cache):
        x = cache
        self.gradient_W = np.dot(x.T, dout)
        self.gradient_b = np.sum(dout, 0)

        dx = np.dot(dout, self.W.T)

        return self.gradient_W, self.gradient_b, dx

    def backward(self, alpha=1e-4, wd=4e-4, momentum = 0.9):
        self.W*=(1-wd)
        self.b*=(1-wd)

        self.weights_m = momentum * self.weights_m - alpha * self.gradient_W
        self.bias_m = momentum * self.bias_m - alpha * self.gradient_b

        self.W += self.weights_m
        self.b += self.bias_m

        self.gradient_W = np.zeros_like(self.gradient_W)
        self.gradient_b = np.zeros_like(self.gradient_b)

if __name__ == '__main__':
    from utils.gradient_check import *


    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    fc = Linear(1024, 10)
    inp = np.random.randn(16,1024)
    weight = np.random.randn(1024,10)
    bias = np.random.randn(10,)

    out, cache = fc.forward(inp)
    d_out = np.random.randn(16,10)
    gradient = fc.gradient(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda w: fc.forward(x=inp, w=w)[0], weight, d_out)
    print(rel_error(dw_num, gradient[0]))


    out, cache = fc.forward(inp)
    d_out = np.random.randn(16,10)
    gradient = fc.gradient(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda x: fc.forward(x=inp)[0], inp, d_out)
    print(rel_error(dw_num, gradient[2]))

    out, cache = fc.forward(inp)
    d_out = np.random.randn(16,10)
    gradient = fc.gradient(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda b: fc.forward(x=inp, b=b)[0], bias, d_out)
    print(rel_error(dw_num, gradient[1]))