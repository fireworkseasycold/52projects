#author:fireworkseasycold
conda create -n 52projects-1 python==3.6.3
conda activate 52projects-1
pip install numpy opencv-python
conda install nb_conda #用于jupyter使用虚拟环境
#终端输入jupyter notebook启动，ctrl+c退出服务
pip install opencv_contrib_python

#3.翻转需要继续安装matplotlib
pip install matplotlib