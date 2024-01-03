

import torch

# 生成初始参数规定范围，
minV = [0, 0]
maxV = [200, 300]

def constrain(t):
    item_num = t.numel()
    if t[0] < 0 or t[1] < 0:
        print("x less zero!")

    for i in range(item_num):
        if t[i] < minV[i]:
            t[i] = maxV[i] - (minV[i] - t[i])
        elif t[i] > maxV[i]:
            t[i] = minV[i] + (t[i] - maxV[i])
    return t

test = torch.tensor([1.0, 1.0])
print(list(range(10)))
