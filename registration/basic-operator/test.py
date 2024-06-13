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

lower_bound = torch.tensor([-5.0, -3.0, -4.0])
upper_bound = torch.tensor([4.0, 2.0, 5.0])

speed = torch.tensor([3.0, 4.0, 8.0])

range_size = upper_bound - lower_bound
position = lower_bound + torch.remainder(speed - lower_bound, range_size)

print(position)

