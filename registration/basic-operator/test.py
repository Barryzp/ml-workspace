import itk
import numpy as np
from glob import glob
import os, torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt


import os, random

import numpy as np

# 创建一个示例数组
arr = np.array([1, 2, 2, 3, 3, 5,5,5,5,5,5,5, 3, 4, 4, 4, 4])

# 使用np.unique获得数组中的元素及其频率
values, counts = np.unique(arr, return_counts=True)

# 使用np.argmax找到最大频率的索引
most_frequent_index = np.argmax(counts)

# 使用该索引获取最频繁的元素
most_frequent_element = values[most_frequent_index]

print("最频繁的元素是:", most_frequent_element)
print("出现次数是:", counts[most_frequent_index])
