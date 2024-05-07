import torch
from torch import nn
import torch.nn.functional as F

class Lenet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5,bias=True) # 3,32,32 -> 6,28,28
        self.pool1=nn.AvgPool2d(2,2) # 6,28,28 -> 6,14,14
        self.conv2=nn.Conv2d(6,16,5,bias=True) #6,14,14 -> 16,10,10
        self.pool2=nn.AvgPool2d(2,2)#pool#16,10,10 -> 16,5,5
        self.conv3=nn.Conv2d(16,120,5,bias=True) #16,5,5 -> 120,1,1
        self.fc1=nn.Linear(120,84,bias=True) #120->84
        self.fc2=nn.Linear(84,10,bias=True) #84 ->10
    def forward(self,x):
        x=F.tanh(self.conv1(x))
        x=self.pool1(x)
        x=F.tanh(self.conv2(x))
        x=self.pool2(x)
        x=F.tanh(self.conv3(x))
        x=x.reshape(-1,120)
        x=F.tanh(self.fc1(x))
        x=self.fc2(x)
        return x