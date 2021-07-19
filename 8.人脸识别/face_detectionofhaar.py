# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/8 16:55
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : face_recognition.py
# @Software: PyCharm
"""
使用opencv自带的haar特征分类器进行人脸定位
"""

import cv2
import random



img=cv2.imread('face.JPG',cv2.IMREAD_COLOR)
# cv2.imshow('img',img)
b, g, r = cv2.split(img)
#单通道提取的图像为灰度图像
# cv2.imshow('b',b)
# cv2.imshow('g',img[:,:,1])
# cv2.imshow('r',img[:,:,2])
img[:,:,2]=0
img[:,:,1]=0
# cv2.imshow('blue',img)

img=cv2.imread('faces.JPG',cv2.IMREAD_GRAYSCALE)
#face,将使用OpenCV中自带的的基于Haar特征的级联分类器进行对象检测
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
faces_detected=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5)
if faces_detected ==():
    ##问题来了，为什么上图修改后单独的蓝色图片不能检测出人脸呢？？？
    print('请确确认图片是否被修改过,例如img[:,:,0]=0,注释掉这几句')
# print(faces_detected)
for i in faces_detected:
    x,y,w,h=i
    #框出face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1) #左上，右下
    # 注意numpy中的坐标是和图片坐标相反的，所以切片要先y，后x,切出face用于深度学习
    face = img[y:y + h, x:x + w]
    cv2.imshow('face',face)
    cv2.waitKey(0)
cv2.imshow('allface',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
