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

        self.patch_num = (image_size//patch_size)**2
        self.input = nn.Linear(channel_size*patch_size**2, d_channel)

        self.layer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(channel_size*patch_size**2, d_channel),
            *[MixerLayer(d_channel, self.patch_num) for _ in range(num_layer)],
            nn.LayerNorm(d_channel),
        )
        self.out = nn.Linear(d_channel, class_num)

    def forward(self, x):
        x = self.layer(x)
        x = torch.mean(x, dim=1)
        x = self.out(x)
        return x
