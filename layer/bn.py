# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 12/1/18
import numpy as np

class BatchNorm2D(object):
    def __init__(self, num_features, affine=True, moving_decay=0.9, epsilon=1e-8):
        '''
        Apply to convolution layer
        :param num_features:
        :param affine:
        :param moving_decay:
        :param epsilon:
        '''
        self.num_features = num_features
        self.affine = affine
        self.epsilon = epsilon
        self.moving_decay = moving_decay
        if self.affine:
            self.weight = np.ones((num_features,))
            self.bias = np.zeros((num_features,))
        self.moving_mean = np.zeros((num_features,))
        self.moving_var = np.ones((num_features,))
        self.mean = np.zeros((num_features,))
        self.var = np.ones((num_features,))
        self.weights_m = np.zeros_like(self.weight)
        self.bias_m = np.zeros_like(self.bias)

    def forward(self, x, training=True):
        assert len(x.shape)==4
        B = x.shape[0]
        mean = np.mean(x, axis=(0,2,3))
        var = np.var(x, axis=(0,2,3))
        self.mean = mean
        self.var = var
        if training:
            if np.mean(self.moving_mean)==0.0 and np.mean(self.moving_var)==1.0:
                self.moving_mean = mean
                self.moving_var = var
            else:
                self.moving_mean = self.moving_mean*self.moving_decay+(1-self.moving_decay)*mean
                self.moving_var = self.moving_var*self.moving_decay+(1-self.moving_decay)*var
            normx = (x - mean[np.newaxis,:,np.newaxis,np.newaxis])/np.sqrt(var[np.newaxis,:,np.newaxis,np.newaxis]+self.epsilon)
        else:
            normx = (x - self.moving_mean[np.newaxis,:,np.newaxis,np.newaxis])/np.sqrt(self.moving_var[np.newaxis,:,np.newaxis,np.newaxis]+self.epsilon)

        cache = (x, normx)

        return normx*self.weight[np.newaxis,:,np.newaxis,np.newaxis]+self.bias[np.newaxis,:,np.newaxis,np.newaxis], cache

    def gradient(self, dout, cache):
        x, normx = cache
        B = x.shape[0]*x.shape[2]*x.shape[3]
        mean = self.mean
        var = self.var
        x_mu = x-mean[np.newaxis,:,np.newaxis,np.newaxis]

        d_normx = dout*self.weight[np.newaxis,:,np.newaxis,np.newaxis]
        d_var = np.sum(d_normx*x_mu, axis=(0,2,3))*-.5*(var+self.epsilon)**(-3/2)
        d_mu = np.sum(d_normx*-1/np.sqrt(var+self.epsilon)[np.newaxis,:,np.newaxis,np.newaxis], axis=(0,2,3))+d_var*np.sum(-2*x_mu,axis=(0,2,3))/B
        d_x = d_normx/np.sqrt(var+self.epsilon)[np.newaxis,:,np.newaxis,np.newaxis]+\
              d_var[np.newaxis,:,np.newaxis,np.newaxis]*2*x_mu/B+\
              d_mu[np.newaxis,:,np.newaxis,np.newaxis]/B
        self.d_weight = np.sum(dout*normx, axis=(0,2,3))
        self.d_bias = np.sum(dout, axis=(0,2,3))

        return self.d_weight, self.d_bias, d_x

    def backward(self, alpha=1e-4, momentum=0.9):
        self.weights_m = momentum * self.weights_m - alpha * self.d_weight
        self.bias_m = momentum * self.bias_m - alpha * self.d_bias
        self.weight += self.weights_m
        self.bias += self.bias_m

        self.d_weight = np.zeros_like(self.d_weight)
        self.d_bias = np.zeros_like(self.d_bias)


class BatchNorm1D(object):
    def __init__(self, num_features, affine=True, moving_decay=0.99, epsilon=1e-8):
        '''
        Apply to the linear layer
        :param num_features:
        :param affine:
        :param moving_decay:
        :param epsilon:
        '''
        self.num_features = num_features
        self.affine = affine
        self.moving_decay = moving_decay
        self.epsilon = epsilon
        if self.affine:
            self.weight = np.ones((num_features,))
            self.bias = np.zeros((num_features,))
        self.moving_mean = np.zeros((num_features,))
        self.moving_var = np.ones((num_features,))
        self.mean = np.zeros((num_features,))
        self.var = np.ones((num_features,))

    def forward(self, x, training=True):
        assert len(x.shape)==2
        B = x.shape[0]
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        self.mean = mean
        self.var = var
        if training:
            if np.mean(self.moving_mean) == 0.0 and np.mean(self.moving_var) == 1.0:
                self.moving_mean = mean
                self.moving_var = var
            else:
                self.moving_mean = self.moving_mean * self.moving_decay + (1 - self.moving_decay) * mean
                self.moving_var = self.moving_var * self.moving_decay + (1 - self.moving_decay) * var
            normx = (x - mean[np.newaxis, :]) / np.sqrt(var[np.newaxis, :] + self.epsilon)
        else:
            normx = (x - self.moving_mean[np.newaxis, :]) / np.sqrt(self.moving_var[np.newaxis, :] + self.epsilon)

        cache = (x, normx)

        return normx * self.weight[np.newaxis, :] + self.bias[np.newaxis, :], cache

    def gradient(self, dout, cache):
        x, normx = cache
        B = x.shape[0]
        mean = self.mean
        var = self.var
        x_mu = x - mean[np.newaxis, :]

        d_normx = dout * self.weight[np.newaxis, :]
        d_var = np.sum(d_normx * x_mu, axis=0) * -.5 * (var + self.epsilon) ** (-3 / 2)
        d_mu = np.sum(d_normx * -1 / np.sqrt(var + self.epsilon)[np.newaxis, :],
                      axis=0) + d_var * np.sum(-2 * x_mu, axis=0) / B
        d_x = d_normx / np.sqrt(var + self.epsilon)[np.newaxis, :] + \
              d_var[np.newaxis, :] * 2 * x_mu / B + \
              d_mu[np.newaxis, :] / B
        self.d_weight = np.sum(dout * normx, axis=0)
        self.d_bias = np.sum(dout, axis=0)

        return self.d_weight, self.d_bias, d_x


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == '__main__':
    from utils.gradient_check import *

    # inp = np.random.randn(2,8,35,35)
    # bn = BatchNorm2D(8)
    # out, cache = bn.forward(inp)
    #
    # dout = np.random.randn(2,8,35,35)
    # gradient = bn.gradient(dout, cache)
    # dx_num = eval_numerical_gradient_array(lambda x: bn.forward(x)[0], inp, dout)
    # print(rel_error(dx_num, gradient[2]))

    inp = np.random.randn(16,1024)
    bn = BatchNorm1D(1024)
    out, cache = bn.forward(inp)

    dout = np.random.randn(16,1024)
    gradient = bn.gradient(dout, cache)
    dx_num = eval_numerical_gradient_array(lambda x: bn.forward(x)[0], inp, dout)
    print(rel_error(dx_num, gradient[2]))
    pass