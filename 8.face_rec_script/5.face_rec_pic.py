# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/6/22 16:12
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : face_recready.py
# @Software: PyCharm

"""
图片的人脸比对,并在图片中写出来结果
"""
import face_recognition
import numpy as np
import cv2



print('开始测试')
# 创建人脸检测级联分类器对象实例
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


image = face_recognition.load_image_file('knownimg/my.jpg')
image2 = face_recognition.load_image_file('knownimg/mht.jpg')
image3 = face_recognition.load_image_file('knownimg/lhy.jpg')
face_encoding1 = face_recognition.face_encodings(image)[0]
# print(len(face_encoding1))
face_encoding2 = face_recognition.face_encodings(image2)[0]
face_encoding3 = face_recognition.face_encodings(image3)[0]
known_face_encodings_dict = {
    'lhy': face_encoding3,
    'mht': face_encoding2,
    'my': face_encoding1,
}
#解压缩，获取key和value各自的元组
known_face_tumple, known_face_encodings_tumple=zip(*known_face_encodings_dict.items())
know_face_encodings_list=list(known_face_encodings_tumple) #元组转列表
# print('加载测试图')
frame=face_recognition.load_image_file('unknownimg/who4.jpg')
# print('定位测试人脸')
face_location4 = face_recognition.face_locations(frame)
# print('打印人脸',face_location4)

face_encoding4 = face_recognition.face_encodings(frame, face_location4)[0]
face_inf=list(zip(face_location4, face_encoding4))
# print(face_inf)

# 对获取的每个人脸进行识别比对
name='未识别'
    # 对其中一个人脸的比对结果（可能比对中人脸库中多个人脸）
for face,a_encoding in known_face_encodings_dict.items():
    #ValueError: too many values to unpack (expected 2
    # 字典只支持Key的遍历,，如果想对key，value，则可以使用items方法。
    #选择1-If a match was found in known_face_encodings_dict, just use the first one.
    matches = face_recognition.compare_faces([a_encoding], face_encoding4,tolerance=0.3)
    if True in matches:
        # print('你好,{}'.format(face))
        name=face
        break
        # engine = pyttsx3.init()
        # engine.say('你好,{}'.format(face))
        # engine.runAndWait()
    else:
        continue
        # engine = pyttsx3.init()
        # engine.say('你是谁，你不是{}'.format(face))
        # engine.runAndWait()
    # 选择2- Or instead, use the known face with the smallest distance to the new face
    # face_distances = face_recognition.face_distance([a_encoding], face_encoding4)
    # if face_distances[0] <= 0.3:
    #     print('相似距离%s' % (face_distances), '你好,{}'.format(face))
    #     result=face
    #     break
    # else:
    #     continue
print(name)

#在图片上显示结果
for top, right, bottom, left in face_location4:
    # 框出人脸
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
    # face = img[y:y + h, x:x + w]
    face = frame[top:bottom, left:right]
    # 在脸下方绘制具有名称的标签
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('face', face[:, :, ::-1])  # 改变channel
    cv2.waitKey(0)  # 注释掉就只显示一张切出的脸

