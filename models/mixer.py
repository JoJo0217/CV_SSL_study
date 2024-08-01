import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MixerLayer(nn.Module):
    def __init__(self, d_channel=512, d_token=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_channel)
        self.norm2 = nn.LayerNorm(d_channel)

        self.token_mlp = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Linear(d_token, d_token)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_channel, d_channel),
            nn.GELU(),
            nn.Linear(d_channel, d_channel)
        )

        nn.init.zeros_(self.token_mlp[-1].weight)
        if self.token_mlp[-1].bias is not None:
            nn.init.zeros_(self.token_mlp[-1].bias)
        nn.init.zeros_(self.channel_mlp[-1].weight)
        if self.channel_mlp[-1].bias is not None:
            nn.init.zeros_(self.channel_mlp[-1].bias)

    def forward(self, x):
        # x shape: (batch, d_token, d_channel)
        residual = x
        x = self.norm1(x)
        x = self.token_mlp(x.transpose(1, 2)).transpose(1, 2)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.channel_mlp(x)
        x = x + residual
        return x


class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, channel_size=3, num_layer=8, d_channel=512, class_num=10):
        super().__init__()
        self.channel_size = channel_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_num = (image_size // patch_size)**2
        self.input = nn.Linear(channel_size * patch_size**2, d_channel)

        self.layer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(channel_size * patch_size**2, d_channel),
            *[MixerLayer(d_channel, self.patch_num) for _ in range(num_layer)],
            nn.LayerNorm(d_channel),
        )
        self.out = nn.Linear(d_channel, class_num)

    def forward(self, x):
        x = self.layer(x)
        x = torch.mean(x, dim=1)
        x = self.out(x)
        return x

    def extract_features(self, x):
        return torch.mean(self.layer(x), dim=1)


class ConvMixerLayer(nn.Module):
    def __init__(self, kernel_size=3, d_channel=512):
        super().__init__()
        self.depthwise = nn.Conv2d(d_channel, d_channel, kernel_size=kernel_size, padding=(
            kernel_size // 2), groups=d_channel)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm2d(d_channel)
        self.pointwise = nn.Conv2d(d_channel, d_channel, kernel_size=1)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(d_channel)

    def forward(self, x):
        residual = x
        x = self.norm1(self.act(self.depthwise(x))) + residual
        # point wise에는 skip connection이 없음
        x = self.norm2(self.act2(self.pointwise(x)))
        return x


class ConvMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=2, channel_size=3, num_layer=8, d_channel=256, class_num=10):
        super().__init__()
        self.channel_size = channel_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_num = (image_size // patch_size)**2
        self.input = nn.Conv2d(channel_size, d_channel,
                               kernel_size=patch_size, stride=patch_size)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(d_channel)
        self.layer = nn.Sequential(
            *[ConvMixerLayer(kernel_size=7, d_channel=d_channel)
              for _ in range(num_layer)],
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flat = nn.Flatten(),  # (batch, d_channel, 1, 1) -> (batch, d_channel)

        self.out = nn.Linear(d_channel, class_num)

    def forward(self, x):
        x = self.bn(self.act(self.input(x)))
        # x shape (batch, d_channel, h/p, w/p)
        x = self.layer(x)
        x = self.out(self.flat(x))
        return x

    def extract_features(self, x):
        x = self.bn(self.act(self.input(x)))
        x = self.layer(x)
        return x
