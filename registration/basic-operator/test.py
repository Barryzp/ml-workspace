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

t = np.array([1, 2, 3, 4, 5])
idx = np.array([0, 2])
print(t[:3])

separator = 10  # 假设separator为10
permuted_sequence = np.random.permutation(separator)

# 从permuted_sequence中随机选择5个数，不放回
selected_numbers = np.random.choice(permuted_sequence, size=5, replace=True)

print("Permuted sequence:", permuted_sequence)
print("Selected numbers:", selected_numbers)


np.random.seed(42)
# 生成一些随机数
random_numbers_1 = np.random.rand(5)
print("Random numbers 1:", random_numbers_1)

# 生成另外一些随机数
random_numbers_2 = np.random.rand(5)
print("Random numbers 2:", random_numbers_2)

print("rand min maxV", np.random.uniform([1, 1], [2, 2]))

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[::3])


print(3 // 2)

