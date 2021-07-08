# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/28 15:13
# @Author : fireworkseasycold
# @Email : 1476094297@qq.com
# @File : Draw _image _saliency_map.py
# @Software: PyCharm
#参考参考链接 https://zhuanlan.zhihu.com/p/115002897
import cv2
import argparse

# construct the argument parser and parse the arguments
def parse_args():
    """ 自定义参数解释器"""
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--image',default='./images/1.jpg',required=False,help='path to inpt image')
    args=ap.parse_args()
    args=vars(args)
    return args

#load the input image
args = parse_args()
image=cv2.imread(args['image'])

#初始化OpenCV的静态显著谱残差检测器
#计算显著图
saliency=cv2.saliency.StaticSaliencySpectralResidual_create()
(success,saliencyMap)=saliency.computeSaliency(image)
saliencyMap=(saliencyMap*255).astype('uint8')

cv2.imshow('image',image)
cv2.imshow('saliencyMap',saliencyMap)

#如果我们想要一个能处理轮廓的二值化图，
#计算凸包，提取边界框等，我们可以用来
#另外，对显著性图进行阈值设置
saliency2 = cv2.saliency.StaticSaliencyFineGrained_create()
(success2, saliencyMap2) = saliency.computeSaliency(image)
saliencyMap2 = (saliencyMap2 * 255).astype("uint8")

threshMap=cv2.threshold(saliencyMap2,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
cv2.imshow('saliencyMap2',saliencyMap2)
cv2.imshow('thresh',threshMap)
cv2.waitKey(0)
cv2.destroyAllWindows()



