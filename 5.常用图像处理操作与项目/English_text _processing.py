# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/8 14:08
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : English_text _processing.py
# @Software: PyCharm
import cv2
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

# image=Image.open('english_word.JPG')
# data=pytesseract.image_to_string(image,'eng')
# print(data)

img=cv2.imread('english_word.JPG')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap='gray')
plt.show()
#模糊处理，消除下划线
output=cv2.medianBlur(gray_img,ksize=5)
plt.imshow(output,cmap='gray')
plt.show()

#侵蚀
kernel=np.ones((3,3),np.uint8)
output2=cv2.dilate(gray_img,kernel,iterations=2)
plt.imshow(output2,cmap='gray')
plt.show()

#膨胀
output3=cv2.erode(gray_img,kernel,iterations=3)
plt.imshow(output3,cmap='gray')
plt.show()

#直方图均衡化
env=cv2.imread('env.JPG',0)
plt.imshow(env,cmap='gray')
plt.show()
hist=cv2.calcHist([env],
                  [0],#使用的通道
                  None,#没有mask,统计整幅图像的直方图，设为None。统计图像某一部分的直方图时，需要掩码图像
                  [256],#histsize
                  [0,255])#直方图柱范围
plt.plot(hist,color='r')
plt.show()

equ=cv2.equalizeHist(env)
plt.imshow(equ,cmap='gray')
plt.show()
hist2=cv2.calcHist([equ],
                   [0],#使用的通道
                  None,#没有mask
                  [256],#histsize
                  [0,255])#直方图柱范)
plt.plot(hist2,color='r')
plt.show()