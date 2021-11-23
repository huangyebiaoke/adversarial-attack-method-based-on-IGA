import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import math,random
import cv2
import matplotlib.pylab as plt
plt.rc('font',family='Times New Roman')

def random_color():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network.load_params()


x_test=np.array(x_test,dtype=np.int)
x_test_ints={
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
}
for i in range(x_test.shape[0]):
    temp=''
    for j in range(x_test.shape[1]):
        temp+=str(x_test[i][j])
    x_test_ints[np.argmax(t_test[i])].append((int(temp,2)/1e224))
# print(len(x_test_string))

losses={
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
}

for i in range(x_test.shape[0]):
    losses[np.argmax(t_test[i])].append(network.loss(x_test[i:i+1], t_test[i:i+1]))


# print(np.mean(x_test_ints[0]),np.median(x_test_ints[0]),np.min(x_test_ints[0]),np.max(x_test_ints[0]))
# print(np.mean(a))

colors={}
for key in losses.keys():
    colors[key]=random_color()

for key in losses.keys():
    plt.scatter(np.full((len(losses[key])),key,'int32'),losses[key],s=8,c=colors[key],label=str(key),alpha=.5)
plt.xlabel('samples of number')
plt.ylabel('loss')
# plt.xlim(1.0391408670466593e+113,5.9391408670466593e+113)
# plt.legend(loc='upper right')
# plt.subplots_adjust(left=0.08, right=1, top=1, bottom=0.09)
# plt.savefig('number_loss.svg')
plt.show()

# cv2.imshow("winname", x_test[2].reshape(28,28))
# cv2.waitKey(0)