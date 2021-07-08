# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/3 16:14
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : coin_recognition.py
# @Software: PyCharm
import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread('coin.jpg')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(thresh,output2)=cv2.threshold(gray_img,120,255,cv2.THRESH_BINARY)
output2=cv2.GaussianBlur(output2,(5,5),1)
output2=cv2.Canny(output2,180,255)
plt.imshow(output2,cmap=plt.get_cmap('gray'))
plt.show()
circles=cv2.HoughCircles(output2,#单通道图像
                         cv2.HOUGH_GRADIENT,#定义检测到圆的方法
                         1, #dp
                         20,#检测到圆的中心
                         param1=50,
                         param2=30,
                         minRadius=0,
                         maxRadius=100)
print(circles)
if circles is None:
    print('未检测到圆，或者请修改霍夫圆检测函数的参数')
else:
    circles=np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2) #画圆
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3) #画圆心
    plt.imshow(img)
    plt.show()