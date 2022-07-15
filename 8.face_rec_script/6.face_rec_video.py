# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/6/16 17:33
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : face_rec.py
# @Software: PyCharm


"""
使用cv2的haar定位人脸
使用face_recognition对视频进行人脸识别
"""
import cv2
import pyttsx3
import face_recognition
import numpy as np

# 创建人脸检测级联分类器对象实例
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')



image1 = face_recognition.load_image_file(r'knownimg/my.jpg')
image2 = face_recognition.load_image_file(r'knownimg/mht.jpg')
image3 = face_recognition.load_image_file(r'knownimg/lhy.jpg')
face_encoding1 = face_recognition.face_encodings(image1)[0]
face_encoding2 = face_recognition.face_encodings(image2)[0]
face_encoding3 = face_recognition.face_encodings(image3)[0]
known_face_encodings_dict = {
    'lhy': face_encoding3,
    'mht': face_encoding2,
    'my': face_encoding1,
}


# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture('mht.mp4')

while camera.isOpened():
    # 参数ret 为True 或者False,代表有没有读取到图片
    # 第二个参数frame表示截取到一帧的图片
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.5, 3)
    for (x, y, w, h) in face:
        # 绘制矩形框，颜色值的顺序为BGR，即矩形的颜色为蓝色
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # 在检测到的人脸区域内检测眼睛
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('video', frame)
    k = cv2.waitKey(1)

    if k == ord('s'): #这里取按下s的当前帧进行识别
        rgb_frame = frame[:, :, ::-1] #bgr与rgb转化

        # 获取画面中的所有人脸位置及人脸特征码

        face_locations = face_recognition.face_locations(rgb_frame)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 对获取的每个人脸进行识别比对

        for (top, right, bottom, left), face_encoding in list(zip(face_locations, face_encodings)):
            # print(face_encoding)
            # 对其中一个人脸的比对结果（可能比对中人脸库中多个人脸）
            for face,a_encoding in known_face_encodings_dict.items():
                # print([np.array(known_face_encodings_dict[face])])
                matches = face_recognition.compare_faces([np.array(known_face_encodings_dict[face])], face_encoding,
                                                         tolerance=0.3)
                if True in matches:
                    #  # If a match was found in known_face_encodings_dict, just use the first one.
                    first_match_index = matches.index(True)
                    print(first_match_index)
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(a_encoding, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    print(best_match_index)
                    if matches[best_match_index]:
                        name = known_face_encodings_dict[best_match_index]

                    engine = pyttsx3.init()
                    engine.say('你好,{}'.format(face))
                    engine.runAndWait()
                else:
                    engine = pyttsx3.init()
                    engine.say('你是谁，你不是{}'.format(face))
                    engine.runAndWait()
                print('---------')
    if k == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
