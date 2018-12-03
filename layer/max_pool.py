# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/30/18
import numpy as np

class MaxPool(object):
    def __init__(self, pool_h, pool_w, stride):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride

    def forward(self, x):
        assert len(x.shape)==4
        (N,C,H,W) = x.shape
        pool_h = self.pool_h
        pool_w = self.pool_w
        stride = self.stride
        H_prime = (H-pool_h)//stride+1
        W_prime = (W-pool_w)//stride+1

        out = np.zeros((N,C,H_prime,W_prime))

        for n in range(N):
            for h in range(H_prime):
                for w in range(W_prime):
                    h1 = h*stride
                    h2 = h1 + pool_h
                    w1 = w*stride
                    w2 = w1 + pool_w
                    window = x[n,:,h1:h2, w1:w2]
                    out[n,:,h,w] = np.max(np.reshape(window, newshape=(C, -1)), axis=1)

        cache = x
        return out, cache

    def gradient(self, dout, cache):
        x = cache
        assert len(x.shape)==4
        (N,C,H,W) = x.shape
        pool_h = self.pool_h
        pool_w = self.pool_w
        stride = self.stride
        H_prime = (H-pool_h)//stride+1
        W_prime = (W-pool_w)//stride+1

        dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h in range(H_prime):
                    for w in range(W_prime):
                        h1 = h*stride
                        h2 = h1+pool_h
                        w1 = w*stride
                        w2 = w1+pool_w
                        window1 = x[n,c,h1:h2, w1:w2].reshape(-1)
                        window2 = np.zeros_like(window1)
                        window2[np.argmax(window1)]=1
                        window3 = window2.reshape(pool_h, pool_w)

                        dx[n,c,h1:h2,w1:w2] = window3*dout[n,c,h,w]

        return dx

if __name__ == '__main__':
    from utils.gradient_check import *


    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    inp = np.random.randn(1, 3, 64, 64)

    dout = np.random.randn(1, 3, 32, 32)

    maxpool = MaxPool(2,2,2)

    out, cache = maxpool.forward(inp)
    gradient = maxpool.gradient(dout, cache)
    dx_num = eval_numerical_gradient_array(lambda x:maxpool.forward(x)[0], inp, dout)
    print(rel_error(dx_num, gradient))
