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

# t[i] = self.minV[i] + t[i] % (self.maxV[i] - self.minV[i])

a = [2,1,8, 9]
print(sorted(a, reverse=True))