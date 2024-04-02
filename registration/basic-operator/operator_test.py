import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from cellpose import models, io
import matplotlib.pyplot as plt
import numpy as np
import cv2


# 初始化CellPose模型
model = models.Cellpose(gpu=False, model_type='cyto')

print("hello")