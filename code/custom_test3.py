import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
import cv2
import numpy as np


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
test_image=np.random.randn(980,980)

cv2.imshow("winname", test_image)
print(x_train[0][10])
cv2.waitKey(0)