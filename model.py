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


class ResBottleNeckBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample=False, stride=None):
        super().__init__()
        if stride is None:
            self.stride = 2 if is_downsample else 1
        else:
            self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        mid_channel = out_channel // 4
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
                      stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if is_downsample:
            self.downsample = self._identitymap
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        return F.relu(self.seq(x) + self.downsample(x))

    def _identitymap(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        dim = self.out_channel//self.in_channel  # in_channel을 out_channel로 만들어야함
        for _ in range(dim//2):  # 2배 -> 1번 4배 -> 2번
            x = torch.concat((x, torch.zeros_like(x)), dim=1)
        return x


class ResBottleNecknet18(nn.Sequential):
    def __init__(self, class_num=10):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBottleNeckBlock(64, 256, is_downsample=True, stride=1),
            ResBottleNeckBlock(256, 256),
            ResBottleNeckBlock(256, 512, is_downsample=True),
            ResBottleNeckBlock(512, 512),
            ResBottleNeckBlock(512, 1024, is_downsample=True),
            ResBottleNeckBlock(1024, 1024),
            ResBottleNeckBlock(1024, 2048, is_downsample=True),
            ResBottleNeckBlock(2048, 2048),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(2048, class_num),
        )


class ResBottleNecknet50(nn.Sequential):
    def __init__(self, class_num=10):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBottleNeckBlock(64, 256, is_downsample=True, stride=1),
            ResBottleNeckBlock(256, 256),
            ResBottleNeckBlock(256, 256),
            ResBottleNeckBlock(256, 512, is_downsample=True),
            ResBottleNeckBlock(512, 512),
            ResBottleNeckBlock(512, 512),
            ResBottleNeckBlock(512, 512),
            ResBottleNeckBlock(512, 1024, is_downsample=True),
            ResBottleNeckBlock(1024, 1024),
            ResBottleNeckBlock(1024, 1024),
            ResBottleNeckBlock(1024, 1024),
            ResBottleNeckBlock(1024, 1024),
            ResBottleNeckBlock(1024, 1024),
            ResBottleNeckBlock(1024, 2048, is_downsample=True),
            ResBottleNeckBlock(2048, 2048),
            ResBottleNeckBlock(2048, 2048),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(2048, class_num),
        )


# preactive paper https://arxiv.org/abs/1603.05027 -> option preactive
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


# resnext paperhttps://arxiv.org/pdf/1611.05431
class ResNextBlock(nn.Module):
    def __init__(self, in_channel, inner_channel, out_channel, cardinality=32, is_downsample=False, stride=None):
        super().__init__()
        if stride is None:
            stride = 2 if is_downsample else 1
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size=3,
                      stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            nn.Conv2d(inner_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        return F.relu(self.seq(x) + self.downsample(x))


class ResNext(nn.Sequential):
    def __init__(self, class_num=10):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),  # 3,32,32 -> 64,32,32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResNextBlock(64, 128, 256, is_downsample=True,
                         stride=1),     # 256,32,32 -> feature size를 유지하고 64를 256으로 증가시키기 위함
            ResNextBlock(256, 128, 256),
            ResNextBlock(256, 256, 512, is_downsample=True),  # 512 16x16
            ResNextBlock(512, 256, 512),
            ResNextBlock(512, 512, 1024, is_downsample=True),  # 1024 8x8
            ResNextBlock(1024, 512, 1024),
            ResNextBlock(1024, 1024, 2048, is_downsample=True),  # 2048 4x4
            ResNextBlock(2048, 1024, 2048),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(2048, class_num),
        )

# attention is all you need paper https://arxiv.org/pdf/1706.03762
class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch, num_head, seq_len, d_model//num_head)
        
        d_k = Q.size(-1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
        #shape: (batch, num_head, seq_len, seq_len)

        if mask is not None:
            attn[mask == 0] = -1e9
        softmax_attn = F.softmax(attn, dim=-1)
        #shape: (batch, num_head, seq_len, seq_len)
        output = torch.matmul(softmax_attn, V)
        #shape: (batch, num_head, seq_len, d_model//num_head)
        return output


class multi_head_Attention(nn.Module):
    def __init__(self, d_model=512, num_head=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k=d_model//num_head
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        # self.Q = nn.Linear(d_model, d_model//num_head)
        # self.K = nn.Linear(d_model, d_model//num_head)
        # self.V = nn.Linear(d_model, d_model//num_head)
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attention = scaled_dot_product_attention()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        # Q, K, V shape: (batch, seq_len, d_model)
        # change into (batch, num_head, seq_len, d_model//num_head)
        Q=Q.view(Q.size(0), Q.size(1), -1, self.d_k).transpose(1, 2)
        K=K.view(K.size(0), K.size(1), -1, self.d_k).transpose(1, 2)
        V=V.view(V.size(0), V.size(1), -1, self.d_k).transpose(1, 2)
        # Q, K, V shape: (batch, num_head, seq_len, d_model//num_head)
        output = self.scaled_dot_product_attention(Q, K, V)
        # output shape: (batch, num_head, seq_len, d_model//num_head)
        output = output.transpose(1, 2).reshape(x.size(0), -1, self.d_model)
        output = self.out(output)
        return output
        
        


MODELS = {
    "lenet5": Lenet5,
    "resnet18": Resnet18,
    "resnet18bottleneck": ResBottleNecknet18,
    "resnet50bottleneck": ResBottleNecknet50,
    "resnet18preact": PreActResNet,
    "resnext": ResNext,
}


def load_model(name, class_num=10):
    if name not in MODELS:
        raise Exception("Unknown model name")
    return MODELS[name](class_num=class_num)
