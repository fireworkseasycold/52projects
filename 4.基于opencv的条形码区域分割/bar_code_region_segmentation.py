# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/30 16:04
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : bar_code_region_segmentation.py
# @Software: PyCharm
import cv2
import matplotlib.pyplot as plt
import  numpy as np
#打开灰度图像
im=cv2.imread('bar.jpg',cv2.IMREAD_GRAYSCALE)
im_out=cv2.imread('bar.jpg')
plt.imshow(im,cmap='gray')



scale=800.0/im.shape[1]
im=cv2.resize(im,(int(im.shape[1]*scale),int(im.shape[0]*scale)))
# blackhat
kernel=np.ones((1,3),np.uint8)
im=cv2.morphologyEx(im,cv2.MORPH_BLACKHAT,kernel,anchor=(1,0))

# 二值化和阈值处理
thresh,im=cv2.threshold(im,10,255,cv2.THRESH_BINARY)


#膨胀和闭合，将上图获取的部分组合在一起成为区域
kernel=np.ones((1,5),np.uint8)
im=cv2.morphologyEx(im,cv2.MORPH_DILATE,kernel,anchor=(2,0),iterations=2)
im=cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel,anchor=(2,0),iterations=2)

#开运算,以删除太少而无法适合条形码形状的与元素
kernel=np.ones((21,35),np.uint8)
im=cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel,iterations=1)


#寻找物体轮廓
contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #只接受二值化数据输入
unscale=1.0/scale
if contours!=None:
    for contour in contours:
        if cv2.contourArea(contour)<=2000:
            continue
        rect=cv2.minAreaRect(contour)
        rect=((int(rect[0][0]*unscale),int(rect[0][1]*unscale)),(int(rect[1][0]*unscale),int(rect[1][1]*unscale)),rect[2])
        box=np.int0(cv2.boxPoints(rect))
        cv2.drawContours(im_out,[box],0,(0,255,0),thickness=2)
plt.imshow(im_out)
plt.show()