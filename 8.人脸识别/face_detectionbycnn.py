# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/19 10:18
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : face_detectionbycnn.py
# @Software: PyCharm
import face_recognition
import cv2

img=face_recognition.load_image_file('knownimg/my.jpg')
# face_locations_hog=face_recognition.face_locations(img)  #[(top,right,bottom,left),...]
# print(face_locations_hog) #(142, 706, 409, 438)
face_locations_cnn=face_recognition.face_locations(img,model='cnn')
print(face_locations_cnn) #(153, 679, 398, 435)

for (top,right,bottom,left) in face_locations_cnn:
    # x=left
    # y=top
    # w=right-left
    # h=bottom-top
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),1)
    # face = img[y:y + h, x:x + w]
    face = img[top:bottom, left:right]
    cv2.imshow('face', face[:,:,::-1])
    cv2.waitKey(0)  # 注释掉就只显示一张切出的脸
cv2.imshow('allface',img[:,:,-1]) #bgr to rgb
cv2.waitKey(0)
cv2.destroyAllWindows()

