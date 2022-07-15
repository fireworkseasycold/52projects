# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/7/15 9:39
# @Author : firworkseasycold
# @Email : 1476094297@qq.com
# @File : views.py
# @Software: PyCharm
from django.http import HttpResponse


def test_ajax_cors(request):
    # return render(request,'test.html') #这么写需要配置settings里templates[]为文件夹'frontend'
    with open('frontend/test.html', 'rb') as f:
        html = f.read()
    return HttpResponse(html)