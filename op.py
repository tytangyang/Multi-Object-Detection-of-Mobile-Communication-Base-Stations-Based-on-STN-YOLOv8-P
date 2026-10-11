import onnx
import torch
import numpy as np
from torch.nn.parameter import Parameter

# 定义PyTorch模型结构
class Model_mnist(torch.nn.Module):
def __init__(self):
super(Model_mnist, self).__init__()
self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
self.conv2 = torch.nn.Conv2d(10, 14, kernel_size=(5, 5), stride=(3, 3), padding=(0, 0))
self.fc1 = torch.nn.Linear(896, 10, bias=True)

def forward(self, x):
out = self.conv1(x)
out = F.relu(out)
out = self.conv2(out)
out = F.relu(out)
out = out.reshape(1, -1)
out = self.fc1(out)
return out

# 加载ONNX模型并提取权重
model_onnx = onnx.load_model("model.onnx")
initializer = model_onnx.graph.initializer

# 创建PyTorch模型并赋值权重
model_torch = Model_mnist()
model_torch.conv1.weight = Parameter(torch.tensor(np.frombuffer(initializer[0].raw_data, dtype=np.float32).reshape(10, 1, 3, 3)))
model_torch.conv1.bias = Parameter(torch.tensor(np.frombuffer(initializer[1].raw_data, dtype=np.float32).reshape(10)))
# ... 继续为其他层赋值

# 保存PyTorch模型
torch.save(model_torch.state_dict(), 'model.pt')