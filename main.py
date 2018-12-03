# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/25/18

import numpy as np
import struct
from glob import glob
from layer.conv import Conv2D
from layer.relu import ReLU
from layer.linear import Linear
from layer.max_pool import MaxPool
from layer.softmax_loss import Softmax_and_Loss
from layer.bn import BatchNorm2D, BatchNorm1D
from layer.dropout import Dropout
from utils.average import AverageMeter

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



if __name__ == '__main__':
    images, labels = load_mnist('./data/mnist')
    test_images, test_labels = load_mnist('./data/mnist', 't10k')
    np.random.seed(2019)

    batch_size = 128
    # Define network
    conv1 = Conv2D(shape=[batch_size, 1, 32, 32], output_channels=6, ksize=5, stride=1, padding=0)    #28*28
    bn1 = BatchNorm2D(6)
    relu1 = ReLU()
    pool1 = MaxPool(2,2,2)  #14*14
    conv2 = Conv2D(shape=[batch_size, 6, 14, 14], output_channels=16, ksize=5, stride=1, padding=0)    #10*10
    bn2 = BatchNorm2D(16)
    relu2 = ReLU()
    pool2 = MaxPool(2,2,2)  # 5*5
    conv3 = Conv2D(shape=[batch_size, 16, 5, 5], output_channels=120, ksize=5, stride=1, padding=0)    #1*1
    bn3 = BatchNorm2D(120)
    relu3 = ReLU()
    fc1 = Linear(1*1*120, 84)
    relu4 = ReLU()
    dp = Dropout(0.9)
    fc2 = Linear(84, 10)
    sf = Softmax_and_Loss()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    init_lr = 0.1
    step=0

    for epoch in range(4):
        # train
        train_acc = 0
        train_loss = 0
        for i in range(images.shape[0] // batch_size):
            step+=1
            lr = init_lr*0.9**(step//50)
            # forward
            img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1]).transpose(0,3,1,2)
            img = np.pad(img, [[0,0],[0,0],[2,2],[2,2]], mode='constant')
            img = img/127.5-1.0
            label = labels[i * batch_size:(i + 1) * batch_size]
            out, conv1_cache = conv1.forward(img)
            out, bn1_cache = bn1.forward(out)
            out, relu1_cache = relu1.forward(out)
            out, pool1_cache = pool1.forward(out)

            out, conv2_cache = conv2.forward(out)
            out, bn2_cache = bn2.forward(out)
            out, relu2_cache = relu2.forward(out)
            out, pool2_cache = pool2.forward(out)

            out, conv3_cache = conv3.forward(out)
            out, bn3_cache = bn3.forward(out)
            out, relu3_cache = relu3.forward(out)

            conv_out = out

            out = conv_out.reshape(batch_size, -1)
            out, fc1_cache = fc1.forward(out)
            out, relu4_cache = relu4.forward(out)
            out, dp_cache = dp.forward(out)

            out, fc2_cache = fc2.forward(out)
            loss, dx =  sf.forward_and_backward(out, np.array(label))

            # calculate gradient
            _, _, dx = fc2.gradient(dx, fc2_cache)

            dx = dp.gradient(dx, dp_cache)
            dx = relu4.gradient(dx, relu4_cache)
            _,_,dx = fc1.gradient(dx, fc1_cache)

            dx = dx.reshape(conv_out.shape)

            dx = relu3.gradient(dx, relu3_cache)
            _,_,dx = bn3.gradient(dx, bn3_cache)
            _,_,dx = conv3.gradient(dx, conv3_cache)


            dx = pool2.gradient(dx, pool2_cache)
            dx = relu2.gradient(dx, relu2_cache)
            _, _, dx = bn2.gradient(dx, bn2_cache)
            _, _ ,dx = conv2.gradient(dx, conv2_cache)

            dx = pool1.gradient(dx, pool1_cache)
            dx = relu1.gradient(dx, relu1_cache)
            _, _, dx = bn1.gradient(dx, bn1_cache)
            _,_, dx = conv1.gradient(dx, conv1_cache)

            conv1.backward(lr)
            bn1.backward(lr)
            conv2.backward(lr)
            bn2.backward(lr)
            conv3.backward(lr)
            bn3.backward(lr)
            fc1.backward(lr)
            fc2.backward(lr)


            print(i,lr, loss)

            if i%20==0:
                val_acc.reset()
                val_loss.reset()
                for k in range(test_images.shape[0] // batch_size):
                    batch_acc = 0
                    img = test_images[k * batch_size:(k + 1) * batch_size].reshape([batch_size, 28, 28, 1]).transpose(0, 3, 1, 2)
                    img = np.pad(img, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
                    img = img / 127.5 - 1.0
                    label = test_labels[k * batch_size:(k + 1) * batch_size]

                    out, conv1_cache = conv1.forward(img)
                    out, bn1_cache = bn1.forward(out, False)
                    out, relu1_cache = relu1.forward(out)
                    out, pool1_cache = pool1.forward(out)

                    out, conv2_cache = conv2.forward(out)
                    out, bn2_cache = bn2.forward(out, False)
                    out, relu2_cache = relu2.forward(out)
                    out, pool2_cache = pool2.forward(out)

                    out, conv3_cache = conv3.forward(out)
                    out, bn3_cache = bn3.forward(out, False)
                    out, relu3_cache = relu3.forward(out)

                    conv_out = out

                    out = conv_out.reshape(batch_size, -1)
                    out, fc1_cache = fc1.forward(out)
                    out, relu4_cache = relu4.forward(out)
                    out = dp.forward(out, False)

                    out, fc2_cache = fc2.forward(out)
                    loss, dx = sf.forward_and_backward(out, np.array(label))

                    pred = np.argmax(out, axis=1)
                    correct = pred.__eq__(label).sum()
                    val_acc.update(correct/label.size*100)
                    val_loss.update(loss)
                print("val acc:", val_acc.avg, "val loss:", val_loss.avg)