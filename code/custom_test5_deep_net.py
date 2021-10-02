# coding: utf-8
from operator import ne
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from ch08.deep_convnet import DeepConvNet
from dataset.mnist import load_mnist
import cv2
from common.functions import softmax


def calculateConfidence(person,label,predict):
    return softmax(predict(person))[0,label]

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("../ch08/deep_convnet_params.pkl")

image=cv2.imread('./paper_images/1.jpg',cv2.IMREAD_GRAYSCALE)

# preprocess
image=np.array(image,dtype=np.float32)
# image=image.reshape(1,-1)
image/=255.
image=np.expand_dims(image,axis=0)
image=np.expand_dims(image,axis=0)
# print(image.shape)

# print(x_train[2:3].shape)

print('confidence:',calculateConfidence(image, 1, network.predict))
# print('confidence:',calculateConfidence(x_train[2:3], t_train[2], network.predict))