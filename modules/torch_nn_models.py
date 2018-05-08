import torch
import torch.nn as nn
from time import process_time, time

class Conv2D(nn.Module):
    def __init__(self):
        super(Conv2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, stride=(2,2))

    def forward(self, x):
        x = self.conv1(x)
        return x

if __name__ == "__main__":
    data = torch.randn(16,3,1024,1024)
    net = Conv2D()
    t1 = process_time()
    for i in range(20):
        output = net(data)
    dt = process_time() - t1
    print(dt)
