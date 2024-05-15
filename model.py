import torch
from torch import nn
import torch.nn.functional as F


class Lenet5(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 6, 5, bias=True),      # 3,32,32 -> 6,28,28
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                 # 6,28,28 -> 6,14,14
            nn.Conv2d(6, 16, 5, bias=True),     #6,14,14 -> 16,10,10
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                 #pool#16,10,10 -> 16,5,5
            nn.Conv2d(16, 120, 5, bias=True),   #16,5,5 -> 120,1,1
            nn.Tanh(),
            nn.Flatten( start_dim=1,end_dim=-1),#120,1,1 -> 120
            nn.Linear(120, 84, bias=True),      #120->84
            nn.Tanh(),
            nn.Linear(84, 10, bias=True),       #84 ->10
            )


class Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample=False, option='A'):
        super().__init__()
        self.is_downsample = is_downsample
        self.option = option
        stride = 2 if is_downsample else 1
        
        #conv2d 기본적으로 he uniform initialization을 사용하기에 따로 설정 x
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, 
                      out_channel, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      padding_mode='zeros', 
                      bias=False,
                      ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,
                      out_channel,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='zeros',
                      bias=False,
                      ),
            nn.BatchNorm2d(out_channel),
            )
        #option B: downsample -> shortcut을 cnn으로 구현
        if self.is_downsample and self.option == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, 
                        out_channel, 
                        kernel_size=1, 
                        stride=stride, 
                        padding=0, 
                        padding_mode='zeros', 
                        bias=False,
                        ),
                nn.BatchNorm2d(out_channel),
                )
    
    def forward(self, x):
        #option A: stride 2로 선택
        if self.is_downsample and self.option == 'A':
            identity_map=torch.zeros((x.size(0), x.size(1)*2, x.size(2)//2, x.size(3)//2), device=x.device)
            identity_map[:, :x.size(1) :, :]=x[:, :, ::2, ::2]
            return F.relu(self.seq(x) + identity_map) 
        elif self.is_downsample and self.option == 'B':
            return F.relu(self.seq(x) + self.downsample(x))
        else:
            return F.relu(self.seq(x) + x)
        

class Resnet18(nn.Sequential):
    def __init__(self, option='A'):
        if option == 'A':
            super().__init__(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), #32x32유지
                nn.BatchNorm2d(64),
                nn.ReLU(),                                                        
                Residual_block(64, 64),                                           #64, 32x32유지      
                Residual_block(64, 64),
                Residual_block(64, 128, is_downsample=True),                      #128, 16x16
                Residual_block(128, 128),
                Residual_block(128, 256, is_downsample=True),                     #256, 8x8
                Residual_block(256, 256),
                Residual_block(256, 512, is_downsample=True),                     #512, 4x4
                Residual_block(512, 512),
                nn.AvgPool2d(4),                                                  #512, 1x1
                nn.Flatten(),
                nn.Linear(512, 10),
                )
        elif option == 'B':
            super().__init__(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), #32x32유지
                nn.BatchNorm2d(64),
                nn.ReLU(),                                                        
                Residual_block(64, 64),                                           #64, 32x32유지      
                Residual_block(64, 64),
                Residual_block(64, 128, is_downsample=True, option='B'),                      #128, 16x16
                Residual_block(128, 128),
                Residual_block(128, 256, is_downsample=True, option='B'),                     #256, 8x8
                Residual_block(256, 256),
                Residual_block(256, 512, is_downsample=True, option='B'),                     #512, 4x4
                Residual_block(512, 512),
                nn.AvgPool2d(4),                                                  #512, 1x1
                nn.Flatten(),
                nn.Linear(512, 10),
                )


def load_model(name):
    if name == "lenet5":
        return Lenet5()
    elif name == "resnet18A":
        return Resnet18('A')
    elif name == "resnet18B":
        return Resnet18('B')
    else:
        raise Exception("Unknown model name")