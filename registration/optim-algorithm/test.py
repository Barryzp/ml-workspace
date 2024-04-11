import numpy as np
import torch
import pandas as pd


t1 = torch.tensor([1,2,3])
t2 = torch.tensor([1,2,3])

print(torch.concat((t1, t2), dim=0).tolist())