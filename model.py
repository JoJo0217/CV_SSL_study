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


# resnet paper https://arxiv.org/abs/1512.03385 -> option A, B
# preactive paper https://arxiv.org/abs/1603.05027 -> option preactive
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample=False):
        super().__init__()
        self.is_downsample = is_downsample
        stride = 2 if is_downsample else 1
        # conv2d 기본적으로 he uniform initialization을 사용하기에 따로 설정 x
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=1, bias=False,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False,),
            nn.BatchNorm2d(out_channel),
        )

        # option B: downsample -> shortcut을 cnn으로 구현
        if self.is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        # option A: stride 2로 선택
        return F.relu(self.seq(x) + self.downsample(x))


class PreActResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample=False):
        super().__init__()
        stride = 2 if is_downsample else 1

        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=1, bias=False,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False,),
        )
        if is_downsample:
            self.downsample = self._identitymap
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        return self.seq(x) + self.downsample(x)
        # identity map을 downsample하는 함수

    def _identitymap(self, x):
        x = x[:, :, ::2, ::2]
        y = torch.zeros_like(x)
        return torch.concat((x, y), dim=1)


class Resnet18(nn.Sequential):
    def __init__(self, class_num=10):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 64),        # 64, 32x32유지
            ResBlock(64, 64),        # 64, 32x32유지
            ResBlock(64, 128, is_downsample=True),                # 128, 16x16
            ResBlock(128, 128),
            ResBlock(128, 256, is_downsample=True),                # 256, 8x8
            ResBlock(256, 256),
            ResBlock(256, 512, is_downsample=True),                # 512, 4x4
            ResBlock(512, 512),
            nn.AvgPool2d(4),                        # 512, 1x1
            nn.Flatten(),
            nn.Linear(512, class_num),
        )


class PreActResNet(nn.Sequential):
    def __init__(self, class_num=10):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            PreActResBlock(64, 64),
            PreActResBlock(64, 64),
            PreActResBlock(64, 128, is_downsample=True),
            PreActResBlock(128, 128),
            PreActResBlock(128, 256, is_downsample=True),
            PreActResBlock(256, 256),
            PreActResBlock(256, 512, is_downsample=True),
            PreActResBlock(512, 512),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512, class_num),
        )


MODELS = {
    "lenet5": Lenet5,
    "resnet18": Resnet18,
    "resnet18preact": PreActResNet,
}


def load_model(name, class_num=10):
    if name not in MODELS:
        raise Exception("Unknown model name")
    return MODELS[name](class_num=class_num)
