import torch

'''
DDN分为几个部分：
    模型结构：
    掩码自回归->全连接->变分层->卷积->全连接->上采样->卷积->上采样->softmaxs
    然后我们经过采样就能得到每一个点的分布


    
'''

