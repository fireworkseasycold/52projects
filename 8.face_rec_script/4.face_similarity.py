# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/19 15:35
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : face_similarity.py
# @Software: PyCharm

"""
使用face_recognition，计算余弦相似度,并对比
"""

import face_recognition
import cv2
import numpy as np


#1.,将图像加载到一个 numpy 数组中
img1=face_recognition.load_image_file('knownimg/lhy.jpg')
img1_2=face_recognition.load_image_file('knownimg/lhy2.jpg')
img1_3=face_recognition.load_image_file('knownimg/lhy3.jpg')
img2=face_recognition.load_image_file('knownimg/mht.jpg')
img3=face_recognition.load_image_file('knownimg/my.jpg')
img3_2=face_recognition.load_image_file('knownimg/my2.jpg')
img4=face_recognition.load_image_file('knownimg/zly.jpg')
img4_2=face_recognition.load_image_file('knownimg/zly2.jpg')
unknownface=face_recognition.load_image_file('unknownimg/who.jpg')
unknownface2=face_recognition.load_image_file('unknownimg/who2.jpg')
unknownface3=face_recognition.load_image_file('unknownimg/who3.jpg')
unknownface4=face_recognition.load_image_file('unknownimg/who4.jpg')

#2.定位，以及将人脸编码成128维float特征向量
face_locations1=face_recognition.face_locations(img1) #获取所有人脸位置
face_encodings1_list=face_recognition.face_encodings(img1, face_locations1) #返回所有人脸编码组成的列表【array[128,],...】
face_encodings1=face_encodings1_list[0]
# print('检测到的人脸数量：',len(face_encodings1_list))
# print('检测到的所有128维特征向量 list：',face_encodings1_list)
# for face_encoding in face_encodings1_list:
#     print("face_encoding len = {} \nencoding:{}\n\n".format(len(face_encoding),face_encoding))
face_locations1_2=face_recognition.face_locations(img1_2)
face_encodings1_2_list=face_recognition.face_encodings(img1_2, face_locations1_2) #返回所有人脸编码组成的列表【array[128,],...】
face_encodings1_2=face_encodings1_2_list[0]

face_locations1_3=face_recognition.face_locations(img1_3)
face_encodings1_3_list=face_recognition.face_encodings(img1_3, face_locations1_3) #返回所有人脸编码组成的列表【array[128,],...】
face_encodings1_3=face_encodings1_3_list[0]


face_locations2=face_recognition.face_locations(img2)
face_encodings2_list=face_recognition.face_encodings(img2, face_locations2)
face_encodings2=face_encodings2_list[0]

face_locations3=face_recognition.face_locations(img3)
face_encodings3_list=face_recognition.face_encodings(img3, face_locations3)
face_encodings3=face_encodings3_list[0]

face_locations3_2=face_recognition.face_locations(img3_2)
face_encodings3_2_list=face_recognition.face_encodings(img3_2, face_locations3_2) #返回所有人脸编码组成的列表【array[128,],...】
face_encodings3_2=face_encodings3_2_list[0]

face_locations4=face_recognition.face_locations(img4)
face_encodings4_list=face_recognition.face_encodings(img4, face_locations4)
face_encodings4=face_encodings4_list[0]

face_locations4_2=face_recognition.face_locations(img4_2)
face_encodings4_2_list=face_recognition.face_encodings(img4_2, face_locations4_2) #返回所有人脸编码组成的列表【array[128,],...】
face_encodings4_2=face_encodings4_2_list[0]
#my
unknownface_locations=face_recognition.face_locations(unknownface)
unknownface_encodings_list=face_recognition.face_encodings(unknownface,unknownface_locations)
unknownface_encodings=unknownface_encodings_list[0]
#zly
unknownface_locations2=face_recognition.face_locations(unknownface2)
unknownface_encodings_list2=face_recognition.face_encodings(unknownface2,unknownface_locations2)
unknownface_encodings2=unknownface_encodings_list2[0]
#lhy
unknownface_locations3=face_recognition.face_locations(unknownface3)
unknownface_encodings_list3=face_recognition.face_encodings(unknownface3,unknownface_locations3)
unknownface_encodings3=unknownface_encodings_list3[0]
#mht
unknownface_locations4=face_recognition.face_locations(unknownface4)
unknownface_encodings_list4=face_recognition.face_encodings(unknownface4,unknownface_locations4)
unknownface_encodings4=unknownface_encodings_list4[0]

#合并所有已知特征向量列表
all_face_encodings=[face_encodings1,face_encodings1_2,face_encodings1_3,face_encodings2,face_encodings3,face_encodings3_2,face_encodings4,face_encodings4_2]
print(all_face_encodings)
print(len(all_face_encodings))



#3.给定一个人脸编码列表，将它们与已知的人脸编码进行比较，距离告诉您面孔的相似程度。越小越相似
#my,对应图5,6
results_list_who=face_recognition.face_distance(all_face_encodings,unknownface_encodings)
print(results_list_who) #[0.73190959 0.70499465 0.65445687 0.64433687 0.42105963 0.41724144 0.93294774 0.91927853]
#zly,对应图7,8
results_list_who2=face_recognition.face_distance(all_face_encodings,unknownface_encodings2)
print(results_list_who2)  #[0.75430415 0.72941424 0.67041979 0.66142024 0.73615765 0.73613938 0.41740735 0.29309082]
#lhy ,对应图1，图2 ，图3
results_list_who3=face_recognition.face_distance(all_face_encodings,unknownface_encodings3)
print(results_list_who3) #[0.23091001 0.34501372 0.29140328 0.58885668 0.67082755 0.65986757 0.75346967 0.74559755]

#对应图4
results_list_who4=face_recognition.face_distance(all_face_encodings, unknownface_encodings4)
print(results_list_who4) #[0.62847778 0.52083143 0.49689526 0.22675983 0.6002956  0.64479535 0.74983203 0.75646831]
index=np.argmin(results_list_who4)
print('----------',index)
#对比人脸编码,根据计算，大概同一个人的在0.3左右
# results_list=list(results_list_who4<=0.3)
# print(results_list)
results_list=face_recognition.compare_faces(all_face_encodings,unknownface_encodings4,tolerance=0.3)


if True in results_list:
    indexofface=results_list.index(True)
    print('mht 第一个识别匹配合格的是索引为{}的人脸'.format(indexofface))