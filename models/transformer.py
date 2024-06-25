import torch
from torch import nn
import torch.nn.functional as F


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