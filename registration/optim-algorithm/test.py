import numpy as np
import torch
import pandas as pd


# 创建一个示例数组
array = np.array([[1, 5, 3], [2, 5, 7]])
array = array.T
df = pd.DataFrame(array, ["x", "y", "z"])
df.to_csv("asdasd.csv")
print(np.insert(array, 0, 100))
