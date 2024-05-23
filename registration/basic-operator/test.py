import itk
import numpy as np
from glob import glob
import os, torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt


import os, random

import numpy as np

# 创建一个示例坐标数组
coordinates = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
])

# 给定点 (x, y)
x, y = 6, 5

# 计算每个坐标点与 (x, y) 的欧氏距离
distances = np.sqrt((coordinates[:, 0] - x)**2 + (coordinates[:, 1] - y)**2)

# 找到最小距离的索引
closest_index = np.argmin(distances)

# 最近点的坐标
closest_point = coordinates[closest_index]

print(f"The closest point to ({x}, {y}) is {closest_point} with index {closest_index}.")