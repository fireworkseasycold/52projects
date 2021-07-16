# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/17 7:19
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : draw_face_key_points.py
# @Software: PyCharm
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/16 8:40
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : easy_face_recognition.py
# @Software: PyCharm

#人脸关键点检测与绘制

import face_recognition
import cv2



# 首先加载图像,将图像加载到一个 numpy 数组中,如果你已经在一个 numpy 数组中有一个图像，你可以跳过这一步。
my_image = face_recognition.load_image_file("knownimg/my.jpg")


# Find all the faces in the image找人脸
my_face_locations = face_recognition.face_locations(my_image)
# print(my_face_locations,len(my_face_locations))#[(142, 706, 409, 438)],top,right,bottom,left

# Or maybe find the facial features in the image识别五官,关键点
my_face_landmarks_list = face_recognition.face_landmarks(my_image)
# print(my_face_landmarks_list)#函数返回一个列表，列表里是个所有关键点的字典
# my_face_landmarks_list=[{
# 'chin': [(442, 246), (444, 276), (448, 307), (455, 338), (468, 367), (490, 390), (520, 408), (552, 423), (585, 424), (618, 418), (648, 402), (673, 380), (688, 353), (695, 321), (698, 291), (699, 261), (700, 232)],'left_eyebrow': [(466, 225), (482, 208), (506, 205), (530, 208), (552, 217)],
# 'right_eyebrow': [(582, 211), (606, 204), (630, 201), (653, 206), (669, 221)],
# 'nose_bridge': [(572, 241), (573, 260), (573, 280), (575, 300)],
# 'nose_tip': [(542, 311), (558, 316), (575, 319), (590, 314), (604, 310)],
# 'left_eye': [(493, 247), (506, 240), (521, 242), (536, 251), (521, 254), (505, 253)],
# 'right_eye': [(602, 248), (614, 237), (630, 236), (644, 243), (632, 248), (617, 250)],
# 'top_lip': [(515, 344), (539, 338), (559, 335), (573, 337), (587, 334), (608, 336), (632, 340), (624, 342), (588, 343), (574, 344), (559, 344), (523, 346)],
# 'bottom_lip': [(632, 340), (611, 357), (592, 364), (576, 365), (561, 365),(541, 360), (515, 344), (523, 346), (560, 351), (575, 352), (590, 351), (624, 342)]
# }]
# 绘制关键点
for part, point_list in my_face_landmarks_list[0].items():
    print(part, point_list)
    for point in point_list:
        cv2.circle(my_image, point, 1, (0, 0, 255), thickness=-1)
cv2.imshow('keyofface', my_image[:,:,::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()