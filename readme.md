# Build a deep neural network from scratch only with Numpy
Nowadays, people can build a deep neural network with the help of the deep learning framework such as TensorFlow, PyTorch, Keras, etc, even though they know little about machine learning and neural networks. For people who want do related work about machine learning, apart from mastering deep learning framework's APIs, knowing how these APIs work is also important. Therefore, in this repo I will try to implement some basic layers (such as conv layer, batch norm layer) only with Numpy. The final goal of this repo is to build a modified lenet and achieve the state-of-the-art result on the MNIST dataset.

# Requirement
- Python3
- Numpy
- Matplotlib

# Dataset
Download MNIST dataset from [Lecun's website](http://yann.lecun.com/exdb/mnist/) and extract it under data/mnist.

# Results
## 2018/12/3
In this version, I have implemented the convolution, linear (fully-connected), batch normalization, dropout, max pooling, ReLU, softmax loss layers. A modified lenet is builded with these layers one by one. We will first forward the network from bottom to top, then calculate every layer's gradient from top to bottom, last we will update the parameters. For now, I have implemented the SGD with momentum. 

![](https://raw.githubusercontent.com/BoyuanJiang/Numpy_CNN/master/fig/1543805683.png)
AS can be seen, we achieve about 99% accuracy on both train and test datasets.

A new version based on compute graph is on the way and I will update it recently.

# Thanks
This work was inspired by [CS231n](http://cs231n.stanford.edu/) and [its assignment](https://github.com/huyouare/CS231n/tree/master/assignment2).