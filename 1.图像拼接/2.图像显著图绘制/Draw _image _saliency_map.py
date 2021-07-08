# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/28 15:13
# @Author : fireworkseasycold
# @Email : 1476094297@qq.com
# @File : Draw _image _saliency_map.py
# @Software: PyCharm

import cv2
import argparse

# construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',default='.',required=True,help='path to inpt image')
args=vars(ap.parse_args())
print(args)
#load the input image
image=cv2.imread(args['image'])

#初始化OpenCV的静态显著谱残差检测器
#计算显著图
saliency=cv2.saliency.StaticSaliencySpectralResidual_create()
(success,saliencyMap)=saliency.computeSaliency(image)
saliencyMap=(saliencyMap*255).astype('uint8')
cv2.imshow('image',image)


