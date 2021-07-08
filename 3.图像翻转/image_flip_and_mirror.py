# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/6/30 12:32
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : image_flip_and_mirror.py
# @Software: PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt

#镜像图像（实质从左到右逐行反转矩阵）
#use a matrix a for example
# a=[[4,4,1],
# #    [2,8,0],
# #    [3,8,1]]
# #
# # mirror_=np.fliplr(a)
# # print(mirror_)
# # #翻转图像
# # flip_=np.flipud(a)
# # print(flip_)

class FlipOrMirror():
    def __init__(self,image_file):
        self.image_file=image_file

    def read_img(self,gray_scale=False):
        img_src = cv2.imread(self.image_file)  # 默认为bgr
        if gray_scale:
            img_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        else:
            img_rgb = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        return img_rgb

    def mirror_img(self,with_plot=True,gray_scale=False):
        image_rgb=self.read_img(gray_scale=gray_scale)
        image_mirror=np.fliplr(image_rgb)
        if with_plot:
            self.plt_img(orig_matrix=image_rgb,trans_matrix=image_mirror,head_text='mirrored')
            return None
        return image_mirror

    def flip_img(self,with_plot=True,gray_scale=False):
        image_rgb=self.read_img(gray_scale=gray_scale)
        image_flip=np.flipud(image_rgb)
        if with_plot:
            self.plt_img(orig_matrix=image_rgb,trans_matrix=image_flip,head_text='fliped')
            return None
        return image_flip

    def plt_img(self,orig_matrix,trans_matrix,head_text,gray_scale=False):
        plt.figure(figsize=(10, 20))
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('off')
        ax1.title.set_text('original')
        ax2 = plt.subplot(2, 2, 2)
        ax2.axis('off')
        ax2.title.set_text(head_text)
        if not gray_scale:
            ax1.imshow(orig_matrix)
            ax2.imshow(trans_matrix)
        else:
            ax1.imshow(orig_matrix,cmap='gray')
            ax2.imshow(trans_matrix,cmap='gray')
        plt.show()
        return True

if __name__ == '__main__':
    fom=FlipOrMirror(image_file='lena.jpg')
    fom.mirror_img()