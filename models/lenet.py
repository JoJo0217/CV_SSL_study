import torch
from torch import nn
import torch.nn.functional as F


class Lenet5(nn.Sequential):
    def __init__(self, class_num=10):
        super().__init__(
            nn.Conv2d(3, 6, 5, bias=True),      # 3,32,32 -> 6,28,28
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                 # 6,28,28 -> 6,14,14
            nn.Conv2d(6, 16, 5, bias=True),     # 6,14,14 -> 16,10,10
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                 # pool#16,10,10 -> 16,5,5
            nn.Conv2d(16, 120, 5, bias=True),   # 16,5,5 -> 120,1,1
            nn.Tanh(),
            nn.Flatten(start_dim=1, end_dim=-1),  # 120,1,1 -> 120
            nn.Linear(120, 84, bias=True),      # 120->84
            nn.Tanh(),
            nn.Linear(84, class_num, bias=True),       # 84 ->10
        )
