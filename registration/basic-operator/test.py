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

# 创建一个示例二值图像
img = np.array([
    [0 ,0, 0, 0, 0],
    [0, 0, 255, 255, 255],
    [0, 0, 255, 0, 255],
    [0, 0, 255, 0, 255],
    [0, 0, 255, 255, 0],
], dtype=np.uint8)

# 检查轮廓的面积
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    print(f"Contour Area: {area}")

# 使用 connectedComponentsWithStats 计算联通区域的面积
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4, ltype=cv2.CV_32S)
for i in range(1, num_labels):  # 从1开始忽略背景
    area = stats[i, cv2.CC_STAT_AREA]
    print(f"Connected Component {i} Area: {area}")
