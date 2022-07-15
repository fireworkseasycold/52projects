import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


def index(request):
    return render(request,'test.html')


#无参数的get的 api（表单提交）
def indexApi(request):
    return JsonResponse({'code':200,'msg':'这是后端jsonresponse','data':'ajax-get不传参'})

#带参数的get的 api（表单提交），参数会 ？拼接ajax参数
def indexApi2(request):
    print(request.GET)
    param=request.GET
    print(param.get('Api2'))
    return JsonResponse({'code':200,'msg':'这是后端jsonresponse','data':'ajax-get传参,参数是：{0}'.format(param.get('Api2'))})



#无参数的post的 api
@csrf_exempt  #防止csrf
def indexApi3(request):
    return JsonResponse({'code':200,'msg':'这是后端jsonresponse','data':'ajax-post不传参'})

#有参数的post的 api
@csrf_exempt  #防止csrf
def indexApi4(request):
    param1=request.body
    print(param1)
    param2=request.POST
    print(param2)
    return JsonResponse({'code':200,'msg':'这是后端jsonresponse','data':'ajax-post传参,参数是：{0}'.format(param2.get('Api4'))})

#表单提交
@csrf_exempt  #防止csrf
def indexApi5(request):
    param=request.POST
    print(param)
    name=param.get('name')
    age=param.get('age')
    return JsonResponse({'code':200,'msg':'ajax post表单','data':'{}今年{}岁'.format(name,age)})

#json格式表单提交，POST无法获取
@csrf_exempt  #防止csrf
def indexApi6(request):
    # print(request.method)
    param_b=request.body
    print(param_b)
    param_dict=json.loads(param_b) #将json格式数据转换为python字典类型,即对对象进行json decode解码
    print(param_dict)
    return JsonResponse({'code':200,'msg':'ajax post json格式提交表单','data':'{}'.format(param_dict)})