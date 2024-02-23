import torch

# 检查是否有可用的GPU
if torch.cuda.is_available():
    # 获取GPU设备数量
    device_count = torch.cuda.device_count()
    
    # 获取当前GPU设备的名称
    current_device = torch.cuda.current_device()
    
    print(f"find {device_count} available.")
    print(f"current device: {torch.cuda.get_device_name(current_device)}")
else:
    print("not find.")

# 创建一个简单的Tensor并将其移到GPU上
tensor = torch.Tensor([1.0, 2.0, 3.0])
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print("Tensor coverted.")

# 创建一个简单的神经网络模型并将其移到GPU上
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3, 2)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
if torch.cuda.is_available():
    model = model.to("cuda")
    print("coverted.")