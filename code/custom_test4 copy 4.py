import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import math
import matplotlib.pylab as plt
from common.functions import softmax

plt.rc('font',family='Times New Roman')


def calculateConfidence(person,label,predict):
    person=np.expand_dims(person,axis=0)
    confidence=softmax(predict(person))[0,np.argmax(label)]
    return confidence

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network.load_params()
test_label=np.array([1,0,0,0,0,0,0,0,0,0])
test_label=np.expand_dims(test_label,axis=0)

images=np.zeros((784,784),dtype=np.float32)
for i in range(images.shape[0]):
    for j in range(images.shape[1]):
        if i==j:
            images[i][j]=1

losses=[]
image_ints=[]
# print(images[:1].shape)
for i in range(images.shape[0]):
    losses.append(calculateConfidence(images[i],test_label,network.predict))
    image_ints.append(math.pow(2,i))

plt.scatter(np.arange(0,784,1),losses,s=8,alpha=.5)
# plt.plot(np.arange(0,784,1),losses)
plt.xlabel('int of image')
plt.ylabel('confidence of 0')
# plt.xlim(1.0391408670466593e+113,5.9391408670466593e+113)
# plt.legend(loc='upper right')
# plt.subplots_adjust(left=0.09, right=1, top=1, bottom=0.09)
# plt.savefig('0_confidence.svg')
plt.show()