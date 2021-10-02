import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import cv2
from GA_test_neuralnet_confidence import calculateConfidence
from common.functions import softmax

def calculateConfidence(person,label,predict):
    person=np.expand_dims(person,axis=0)
    confidence=softmax(predict(person))[0,np.argmax(label)]
    return confidence

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network.load_params()

# for i in range(3000):
#     if(np.argmax(t_train[i])==1):
#         cv2.imwrite('./exp_images/1/'+str(i)+'.jpg',x_train[i].reshape(28,28)*255)

image=cv2.imread('./paper_images/NN/0.jpg',cv2.IMREAD_GRAYSCALE)

# preprocess
image=np.array(image,dtype=np.float32)
image=image.reshape(1,-1)
image/=255.


test_label=np.array([1,0,0,0,0,0,0,0,0,0])
test_label=np.expand_dims(test_label,axis=0)

print('confidence:',calculateConfidence(image,test_label,network.predict),'loss:',network.loss(image, test_label))