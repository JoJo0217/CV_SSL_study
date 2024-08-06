import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.model import load_model
from utils.optim import load_optimizer, load_criterion, load_scheduler
from utils.dataset import load_data


class MoCo(nn.Module):
    def __init__(self, device, args, dim=128, queue_size=65536, m=0.999, tau=0.07):
        super().__init__()
        self.dim = dim
        self.m = m
        self.tau = tau
        self.queue_size = queue_size
        self.encoder = load_model(args.model, class_num=dim)
        self.encoder = self.encoder.to(device)
        self.key_encoder = load_model(args.model, class_num=dim)
        self.key_encoder = self.key_encoder.to(device)

        # in the paper moco use encoder with average pool layer as output
        dim_mlp = self.encoder.out.weight.shape[1]

        self.key_encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim),
        )
        self.encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim),
        )
        for param_q, param_k in zip(self.encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, query, key):
        # X: N, C, H, W
        query = self.encoder(query)
        query = nn.functional.normalize(query, dim=1)

        with torch.no_grad():
            self.update_key()
            key = self.key_encoder(key)
            key = nn.functional.normalize(key, dim=1)
        # (N, 128)

        l_pos = torch.bmm(query.view(query.size(0), 1, -1),
                          key.view(key.size(0), -1, 1)).squeeze(-1)  # (N,1,128) (N,128,1) -> (N,1,1) -> (N,1)
        # (N,1)
        l_neg = torch.mm(query, self.queue.clone().detach())
        # (N,128) (128,queue_size) -> (N,queue_size)

        # (N,1)+(N,queue_size) -> (N,queue_size+1)
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.tau
        labels = torch.zeros(logits.size(0)).long().to(
            logits.device)  # 0번이 positive니까

        self.dequeue_and_enqueue(key)
        return logits, labels

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_ptr)
        # self.queue[:, ptr:ptr + batch_size].data = keys.T  <- error 복사가 이루어지지 않음
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_key(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def save_model(self, output):
        torch.save(self.encoder, output)


class RotNet(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.encoder = load_model(args.model, class_num=4)
        self.encoder = self.encoder.to(device)

        dim_mlp = self.encoder.out.weight.shape[1]
        self.encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 4),
        )

    def forward(self, x, _):
        # (batch, 3, 32, 32)
        with torch.no_grad():
            label = torch.zeros(x.size(0) * 4, dtype=torch.long)
            for i in range(4):
                label[x.size(0) * i:x.size(0) * (i + 1)] = i

            label = label.to(x.device)
            arr = []
            for i in range(4):
                arr.append(torch.rot90(x, k=i, dims=(2, 3)))
            arr = torch.concat(arr, dim=0)
        # (batch*4, 3, 32, 32)
        output = self.encoder(arr)
        # (batch*4, 4)
        return output, label


class Simclr(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        self.encoder = load_model(args.model, class_num=128)
        self.encoder = self.encoder.to(device)
        self.key_encoder = load_model(args.model, class_num=128)
        self.key_encoder = self.key_encoder.to(device)

        dim_mlp = self.encoder.out.weight.shape[1]
        self.encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 128),
        )
        self.key_encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 128),
        )

    def forward(self, query, key):
        # query, key -> (batch, 3, 32, 32)
        query = self.encoder(query)
        query = nn.functional.normalize(query, dim=1)

        with torch.no_grad():
            key = self.key_encoder(key)
            key = nn.functional.normalize(key, dim=1)

        # (2*batch, 128) query-key 쌍 생성
        logit = torch.cat([query, key], dim=0)
        # (2*batch,2*batch)
        logit = torch.mm(logit, logit.T)
        for i in range(query.size(0) * 2):
            logit[i, i] = -1e9
        # 자기꺼는 제외

        label = torch.arange(2 * query.size(0)).to(query.device)
        for i in range(query.size(0)):
            label[i + query.size(0)] = i
            label[i] = i + query.size(0)
        # (2*batch, batch)가 나옴
        return logit, label


TRAINERS = {
    "rotnet": RotNet,
    "moco": MoCo,
    "simclr": Simclr,
}


def load_trainer(args, device):
    if args.pretrain not in TRAINERS:
        raise Exception("Unknown pretrain method")
    else:
        return TRAINERS[args.pretrain](device, args)
