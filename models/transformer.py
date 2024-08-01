import torch
from torch import nn
import torch.nn.functional as F


# attention is all you need paper https://arxiv.org/pdf/1706.03762
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch, num_head, seq_len, d_model//num_head)

        d_k = Q.size(-1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
        # shape: (batch, num_head, seq_len, seq_len)

        if mask is not None:
            attn[mask == 0] = -1e9
        softmax_attn = F.softmax(attn, dim=-1)
        # shape: (batch, num_head, seq_len, seq_len)
        output = torch.matmul(softmax_attn, V)
        # shape: (batch, num_head, seq_len, d_model//num_head)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_head
        self.num_head = num_head

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.out = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        # Q, K, V shape: (batch, seq_len, d_model)
        # change into (batch, num_head, seq_len, d_model//num_head)
        Q = Q.view(Q.size(0), Q.size(1), -1, self.d_k).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), -1, self.d_k).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), -1, self.d_k).transpose(1, 2)
        # Q, K, V shape: (batch, num_head, seq_len, d_model//num_head)
        output = self.ScaledDotProductAttention(Q, K, V)
        # output shape: (batch, num_head, seq_len, d_model//num_head)
        output = output.transpose(1, 2).reshape(x.size(0), -1, self.d_model)
        output = self.out(output)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.zeros_(self.FFN[-1].weight)
        if self.FFN[-1].bias is not None:
            nn.init.zeros_(self.FFN[-1].bias)

    def forward(self, x):
        x = x + self.dropout1(self.norm1(self.attention(x)))
        x = x + self.dropout2(self.norm2(self.FFN(x)))
        return x


# ViT paper https://arxiv.org/pdf/2010.11929
class ViT(nn.Module):
    def __init__(self, class_num=10, d_model=384, num_head=6, img_size=32, patch_size=4, num_block=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patch = (img_size // patch_size)**2  # 64개
        self.d_model = d_model
        self.num_head = num_head  # 각 head마다 d_k=d_v=64

        self.layers = nn.ModuleList([EncoderBlock(d_model, num_head)
                                     for _ in range(num_block)])
        self.out = nn.Linear(d_model, class_num)

        # input shape: (batch, 3, 32, 32)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.patch_embedding = nn.Linear(3 * patch_size**2, d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patch + 1, d_model))

        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, x):
        # input shape: (batch, 3, 32, 32)
        # 이미지를 patch로 나누기
        # (batch, 3, 32, 32) -> (batch, 3, 8, 8, 4, 4)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        # (batch, 3, 8, 8, 4, 4) -> (batch, 8, 8, 3, 4, 4) -> (batch, 64, 3*4*4)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(
            x.size(0), -1, self.patch_size**2 * 3)
        # (batch, 64, 3*4*4) -> (batch, 65, 3*4*4) cls_token 추가
        cls_token = self.cls_token.expand(x.size(0), 1, self.d_model)
        x = torch.cat((cls_token, self.patch_embedding(x)), dim=1)
        x = x + self.pos_embedding
        # (batch, 65, 384)

        for layer in self.layers:
            x = layer(x)

        x = x[:, 0]
        return self.out(x)
