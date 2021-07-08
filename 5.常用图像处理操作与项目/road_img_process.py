# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/30 21:50
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : img_process.py
# @Software: PyCharm
import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
#读取图像
img=cv2.imread('road.jpg')
#注意cv读时bgr通道，plt是rgb,还有pltimshow默认使用三通道显示图像需要指定cmap='gray'才是灰度图5
print(img.shape)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

#转换为灰度图
gray_image=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image,cmap='gray')
plt.show()
print(gray_image.shape)

#二值化
# thresh,blackAndWhiteimage=cv2.threshold(gray_image,20,255,cv2.THRESH_BINARY)
# thresh,blackAndWhiteimage=cv2.threshold(gray_image,80,255,cv2.THRESH_OTSU)
thresh,blackAndWhiteimage=cv2.threshold(gray_image,80,255,cv2.THRESH_BINARY)
print(thresh)
plt.imshow(blackAndWhiteimage,cmap='gray')
plt.show()

#图像模糊
blur_image=cv2.blur(gray_image,(10,10))
plt.imshow(blur_image,cmap='gray')
plt.show()
gaussianBlurImage=cv2.GaussianBlur(gray_image,(9,9),5)
plt.imshow(gaussianBlurImage,cmap='gray')
plt.show()

#图像旋转（经常用以数据扩充）
h,w=img_rgb.shape[:2]
center=(w/2,h/2)
M=cv2.getRotationMatrix2D(center,13,scale=1.1)#中心值，角度，缩放比例
rotated=cv2.warpAffine(gray_image,M,(w,h)) #仿射变换
plt.imshow(rotated,cmap='gray')
plt.show()

#边缘检测
#使用canny,只保留路径和车道，用于自动驾驶的数据
t,pic=cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)
pic_gauss=cv2.GaussianBlur(pic,(5,5),3)
cannyedpic=cv2.Canny(pic_gauss,180,255)
plt.imshow(cannyedpic,cmap='gray')
plt.show()
#如果未经过GaussianBlur，则获得的边缘会有更多噪声
cannyedpic_nogauss=cv2.Canny(pic,180,255)
plt.imshow(cannyedpic_nogauss,cmap='gray')
plt.show()

#基于确定的边缘在真实图像上进行处理
lines=cv2.HoughLinesP(cannyedpic,1,np.pi/180,30)
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img_rgb,(x1,y1),(x2,y2),(0,255,0),4)
plt.imshow(img_rgb)
plt.show()

#使用掩蔽方法来处理被视为道路边界的云
def mask_of_image(image):
    height=image.shape[0]
    polygons=np.array([[(0,height),(2200,height),(250,100)]])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

img_rgb_masked=mask_of_image(img_rgb)
lines=cv2.HoughLinesP(cannyedpic,1,np.pi/180,30)
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img_rgb_masked,(x1,y1),(x2,y2),(0,255,0),4)
plt.imshow(img_rgb_masked)
plt.show()
