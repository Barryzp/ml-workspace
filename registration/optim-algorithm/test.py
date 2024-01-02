import torch


t1 = torch.rand([2])
t2 = torch.rand([2])

t = torch.stack([t1, t2])

t = torch.cat([t, t1.unsqueeze(0)], dim=0)
print(t)