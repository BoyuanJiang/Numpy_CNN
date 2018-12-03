# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/30/18

import numpy as np

class Softmax_and_Loss(object):
    """
    Compute the softmax and loss together
    """
    def forward_and_backward(self, x, y):
        """
        forward and backward
        :param x: final layer output, before softmax layer, [N,C]
        :param y: ground truth, [N,]
        :return:
        """
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N),y]))/N
        dx = probs.copy()
        dx[np.arange(N), y]-=1.0

        dx/=N   # for the loss is divided by N
        return loss, dx

if __name__ == '__main__':
    from utils.gradient_check import *


    def rel_error(x, y):
        """ returns relative error """
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    inp = np.random.randn(32,10)
    target = np.ones(32, dtype=int)
    dout = np.ones_like(inp)

    softmax = Softmax_and_Loss()
    gradient = softmax.forward_and_backward(inp, target)[1]
    dx_num = eval_numerical_gradient_array(lambda x: softmax.forward_and_backward(x, target)[0], inp, 1)

    print(rel_error(dx_num, gradient))
