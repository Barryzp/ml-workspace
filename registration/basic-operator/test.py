import itk, cv2
import numpy as np
from glob import glob
import os, torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt


import os, random

import numpy as np

import cv2
import numpy as np

X = np.array([[1, 2, 3], 
              [4, 5, 6]])
class BatchNorm:
    def __init__(self, D):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
    
    def __call__(self, X):
        mean = X.mean(axis=0)
        variance = X.var(axis=0)
        X_hat = (X - mean) / np.sqrt(variance + 1e-8)
        return self.gamma * X_hat + self.beta

# Initialize BatchNorm with 3 features
batch_norm = BatchNorm(3)

# Apply BatchNorm
output_bn = batch_norm(X)
print("BatchNorm output:\n", output_bn)

class LayerNorm:
    def __init__(self, D):
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
    
    def __call__(self, X):
        mean = X.mean(axis=1, keepdims=True)
        variance = X.var(axis=1, keepdims=True)
        X_hat = (X - mean) / np.sqrt(variance + 1e-8)
        return self.gamma * X_hat + self.beta

# Initialize LayerNorm with 3 features
layer_norm = LayerNorm(3)

# Apply LayerNorm
output_ln = layer_norm(X)
print("LayerNorm output:\n", output_ln)