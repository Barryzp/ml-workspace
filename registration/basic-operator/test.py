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


test_indeces = np.array([1, 3, 4, 8, 7, 5])
layer_size = len(test_indeces)
random_indices = np.random.permutation(layer_size) # 生成一个随机排序的序列
separator = layer_size // 2
random_pairs = np.column_stack((random_indices[:separator], random_indices[separator:2 * separator]))

# 获取 test_indeces 中的元素对
first_elements = test_indeces[random_pairs[:, 0]]
second_elements = test_indeces[random_pairs[:, 1]]

comparison_mask = (test_indeces[random_pairs[:, 0]] > test_indeces[random_pairs[:, 1]])


print(random_pairs)
print(comparison_mask)
print(np.where(comparison_mask, random_pairs[:, 0], random_pairs[:, 1]))

print(test_indeces[np.random.permutation(8) % 4])