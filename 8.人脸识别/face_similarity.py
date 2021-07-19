# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/19 15:35
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : face_similarity.py
# @Software: PyCharm
"""
使用face_recognition，计算欧式距离，进行人脸相似度对比
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/19 14:11
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : compareface.py
# @Software: PyCharm
"""
比较人脸相似度，进行识别
"""
import face_recognition
import cv2

img1=face_recognition.load_image_file('knownimg/my.jpg')
img2=face_recognition.load_image_file('knownimg/mht.jpg')
img3=face_recognition.load_image_file('knownimg/me.jpg')
unknownface=face_recognition.load_image_file('unknownimg/unknown1.jpeg')

face_locations1=face_recognition.face_locations(img1)
face_encodings1=face_recognition.face_encodings(img1, face_locations1) #返回所有人脸编码组成的列表【array[],...】,每个是128特征
# for face_encoding in face_encodings1:
#     print("face_encoding len = {} \nencoding:{}\n\n".format(len(face_encoding),face_encoding))
face_locations2=face_recognition.face_locations(img2)
face_encodings2=face_recognition.face_encodings(img2, face_locations2)
face_locations3=face_recognition.face_locations(img3)
face_encodings3=face_recognition.face_encodings(img3, face_locations3)

unknownface_locations=face_recognition.face_locations(unknownface)
unknownface_encodings=face_recognition.face_encodings(unknownface,unknownface_locations)

#合并所有已知特征向量列表
face_encodings=face_encodings1+face_encodings2+face_encodings3
print(face_encodings)

results_list=face_recognition.face_distance(face_encodings,unknownface_encodings[0])
print(results_list) #[0.42105963 0.64433687 0.63999646]
