import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import math,random
import cv2
import matplotlib.pylab as plt
plt.rc('font',family='Times New Roman')

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network.load_params()
test_label=np.array([1,0,0,0,0,0,0,0,0,0])
test_label=np.expand_dims(test_label,axis=0)


# TODO:画出连续0的连续plot图
images=np.zeros((784,784),dtype=np.float32)
for i in range(images.shape[0]):
    for j in range(images.shape[1]):
        if i==j:
            images[i][j]=1

losses=[]
image_ints=[]
# print(images[:1].shape)
for i in range(images.shape[0]):
    losses.append(network.loss(images[i:i+1],test_label))
    image_ints.append(math.pow(2,i))

# plt.scatter(np.arange(0,784,1),losses,s=8,alpha=.5)
plt.plot(np.arange(5.197086986566064e+148,5.197086986566064e+148,1),losses)
plt.xlabel('image_int')
plt.ylabel('loss')
# plt.xlim(1.0391408670466593e+113,5.9391408670466593e+113)
# plt.legend(loc='upper right')
plt.show()