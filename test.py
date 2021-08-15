# test.py

from python_dpcpp_module import image_convolution
import numpy as np
import cv2

image_file = './resources/car_1.bmp'

kernel=np.array([[3,3,3],
                 [-3,-3,-3],
                 [0,0,0]], dtype=np.float32)

#kernel=np.array([[1,1,1],
#                 [1,1,1],
#                 [1,1,1]], dtype=np.float32)

img_in = cv2.imread(image_file)
img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY);
img_ch = cv2.split(img_in)
cv2.imshow('input', img_ch[0])
img_out=image_convolution(img_ch[0], kernel)
cv2.imshow('output', img_out)
cv2.waitKey(0)
print('hello')
