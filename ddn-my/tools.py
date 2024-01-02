import torch
import torch.nn as nn


# 通过vl层得到映射到正态分布上
class VariationlLayer(nn.Module):
    def __init__(self, input_dim, ouput_dim) -> None:
        super(VariationlLayer).__init__()
        self.fl_mu = nn.Linear(in_features=input_dim, out_features=ouput_dim)
        # https://imathworks.com/cv/solved-why-we-learn-logsigma2-in-vae-reparameterization-trick-instead-of-standard-deviation/
        # it brings stability and ease of training. by definition sigma has to be a positive real number. one way to enforce this would be to use a ReLU funtion to obtain its value, but the gradient is not well defined around zero. in addition, the standard deviation values are usually very small 1>>sigma>0. the optimization has to work with very small numbers, where the floating point arithmetic and the poorly defined gradient bring numerical instabilities.
        # 惯用方法（不知道为啥这样，或许是为了优化），因为学习出来的σ^2是正值，
        # 相当于多加了一层约束，难学了，不如就这样，之后求个指数就行。
        self.fl_log_sigma = nn.Linear(in_features=input_dim, out_features=ouput_dim)
        # epsilo
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0