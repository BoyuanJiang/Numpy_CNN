# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/23/18
import  numpy as np
import math
from functools import reduce

class Conv2D(object):
    def __init__(self, shape, output_channels, ksize, stride=1, padding=1, weights=None):
        assert len(shape)==4
        assert isinstance(ksize, int) or len(ksize)==2
        self.inp_shape = shape
        self.output_channels = output_channels
        self.padding = padding
        self.input_channels = shape[1]
        self.input_height = shape[2]
        self.input_width = shape[3]
        self.bs = shape[0]
        self.stride = stride
        self.ksize = (ksize,ksize) if isinstance(ksize, int) else ksize
        assert (self.input_height-self.ksize[0]+2*padding)%stride==0
        assert (self.input_width-self.ksize[1]+2*padding)%stride==0
        self.output_height = (self.input_height-self.ksize[0]+2*padding)//stride+1
        self.output_width = (self.input_width-self.ksize[1]+2*padding)//stride+1

        # self.weights = np.random.standard_normal((self.output_channels, self.input_channels,
        #                                           ksize[0], ksize[1]))
        # self.bias = np.random.standard_normal(self.output_channels)
        fan_in = self.input_channels*self.ksize[0]*self.ksize[1]
        self.weights = np.random.standard_normal((self.output_channels, self.input_channels, self.ksize[0], self.ksize[1]))/np.sqrt(fan_in/2)
        self.bias= np.zeros(self.output_channels)

        self.delta = np.zeros((self.bs, self.output_channels, self.output_height, self.output_width))

        self.w_gradient = np.zeros_like(self.weights)
        self.b_gradient = np.zeros_like(self.bias)
        self.weights_m = np.zeros_like(self.weights)
        self.bias_m = np.zeros_like(self.bias)

        self.output_shape = self.delta.shape
        self.i, self.j, self.k = self.get_im2col_slice(x_shape=shape, ksize=self.ksize)

    def forward(self, x, w=None, b=None):
        '''
        convolution forward
        :param x: input, shape is [batch_size, channels, height, width]
        :return:
        '''
        if w is not None:
            self.weights = w
        if b is not None:
            self.bias = b
        weight_cols = self.weights.reshape(self.output_channels, -1)  # 64,27
        self.x_pad = x
        if self.padding!=0:
            self.x_pad = np.pad(x, ((0,0), (0,0),(self.padding, self.padding),
                                    (self.padding, self.padding)), mode="constant")
        img_cols = self.x_pad[:,self.k, self.i, self.j]
        img_cols = img_cols.transpose(1,0,2)
        img_cols = img_cols.reshape(img_cols.shape[0], -1)  # 27,50176
        out = np.dot(weight_cols, img_cols).transpose(1,0)+self.bias    # 50176,64
        out = out.reshape(self.bs, self.output_height, self.output_width, -1).transpose(0,3,1,2)

        cache = (x, self.weights, self.bias, img_cols)
        return out, cache

    def gradient(self, delta, cache):
        x, self.weights, self.bias, img_cols = cache
        stride = self.stride
        padding = self.padding

        delta_reshape = delta.transpose(1,0,2,3)
        delta_reshape = delta_reshape.reshape(delta_reshape.shape[0], -1)
        self.w_gradient = np.dot(delta_reshape, img_cols.T).reshape(self.w_gradient.shape)
        self.b_gradient = np.sum(delta_reshape,axis=1)

        # backward to get the previous layer's delta
        num_filters, _, f_h, f_w = self.weights.shape
        dx_cols = self.weights.reshape(num_filters, -1).T.dot(delta_reshape)
        dx = self.col2im(dx_cols, x.shape, ksize=self.ksize)

        # delta_pad = delta
        # if self.padding!=0:
        #     delta_pad = np.pad(delta, ((0,0), (0,0), (self.padding, self.padding),
        #                        (self.padding, self.padding)), mode="constant")
        # delta_cols = delta_pad[:,self.k, self.i, self.j]    # 4,27,50176
        # delta_cols = delta_cols.transpose(1,0,2)
        # delta_cols = delta_cols.reshape(delta_cols.shape[0], -1) #27,200704
        # fliped_weight = np.flip(np.flip(self.weights, axis=2), axis=3)
        # fliped_weight_cols = fliped_weight.reshape(self.output_channels, -1)  # 64,27
        # prev_delta = np.dot(fliped_weight_cols, delta_cols)    # 64,200704
        # prev_delta = prev_delta.reshape()

        return self.w_gradient, self.b_gradient, dx


    def get_im2col_slice(self, x_shape, ksize):
        k_h, k_w = ksize
        bs, c, i_h, i_w = x_shape
        i0 = np.repeat(np.arange(k_h), k_w)
        i0 = np.tile(i0, c)
        i1 = self.stride*np.repeat(np.arange(self.output_height), self.output_width)
        i= i0.reshape(-1,1)+i1.reshape(1,-1)

        j0 = np.tile(np.arange(k_w), k_h*c)
        j1 = self.stride*np.tile(np.arange(self.output_width), self.output_height)
        j=j0.reshape(-1,1)+j1.reshape(1,-1)

        k = np.repeat(np.arange(c), k_w*k_h).reshape(-1,1)

        return i,j,k

    def col2im(self, cols, x_shape, ksize):

        padding = self.padding
        N, C, H, W = x_shape
        field_height, field_width = ksize
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

        i,j,k = self.get_im2col_slice(x_shape, (field_height, field_width))
        cols_reshaped = cols.T.reshape(N, C * field_height * field_width, -1)
        # cols_reshaped = cols_reshaped.transpose(2, 1, 0)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded

        return x_padded[:, :, padding:-padding, padding:-padding]

    def backward(self, alpha=1e-4, wd=4e-4, momentum=0.9):
        self.weights *= (1 - wd)
        self.bias *= (1 - wd)
        self.weights_m = momentum*self.weights_m-alpha*self.w_gradient
        self.bias_m = momentum*self.bias_m-alpha*self.b_gradient
        self.weights += self.weights_m
        self.bias += self.bias_m

        self.w_gradient = np.zeros_like(self.w_gradient)
        self.b_gradient = np.zeros_like(self.b_gradient)


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == '__main__':
    from utils.gradient_check import *
    # import torch
    # import torch.nn.functional as F
    # inp_t = torch.randn((4, 3, 224, 224))
    # filter_t = torch.randn((64, 3, 3, 3))
    # out_t = F.conv2d(inp_t, filter_t, stride=1, padding=1)

    # inp = inp_t.numpy()
    # filter = filter_t.numpy()
    inp = np.arange(1,1+2*1*3*3).reshape(2,1,3,3)*1.0
    weight = np.arange(1, 1 + 1 * 1 * 2 * 2).reshape(1, 1, 2, 2) * 1.0
    conv = Conv2D(shape=inp.shape, output_channels=1, ksize=2, stride=1, padding=0)
    out, cache = conv.forward(inp)
    d_out = np.arange(1,1+2*1*2*2).reshape(2,1,2,2)*1.0
    gradient = conv.gradient(d_out, cache)
    dw_num = eval_numerical_gradient_array(lambda w: conv.forward(x=inp, w=w)[0], weight, d_out)
    print(rel_error(dw_num, gradient[0]))

    conv = Conv2D(shape=inp.shape, output_channels=1, ksize=2, stride=1, padding=0)
    out, cache = conv.forward(inp)
    d_out = np.arange(1, 1 + 2 * 1 * 2 * 2).reshape(2, 1, 2, 2) * 1.0
    gradient = conv.gradient(d_out, cache)
    dx_num = eval_numerical_gradient_array(lambda x: conv.forward(x)[0], inp, d_out)
    print(rel_error(dx_num, gradient[2]))


    bias = np.random.randn(1,)
    conv = Conv2D(shape=inp.shape, output_channels=1, ksize=2, stride=1, padding=0)
    out, cache = conv.forward(inp)
    d_out = np.arange(1, 1 + 2 * 1 * 2 * 2).reshape(2, 1, 2, 2) * 1.0
    gradient = conv.gradient(d_out, cache)
    dx_num = eval_numerical_gradient_array(lambda bias: conv.forward(x=inp, b=bias)[0], bias, d_out)
    print(rel_error(dx_num, gradient[1]))
