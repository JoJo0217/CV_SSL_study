import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.model import load_model
from utils.train import Framework


class MoCo(Framework):
    def __init__(self, device, args, dim=128, queue_size=65536, m=0.999, tau=0.07):
        self.dim = dim
        self.m = m
        self.tau = tau
        self.queue_size = queue_size

        model = load_model(args.model, class_num=dim)
        super().__init__(model, criterion=nn.CrossEntropyLoss(), device=device)
        self.key_encoder = copy.deepcopy(self.encoder)

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
        self.encoder = self.encoder.to(device)
        self.key_encoder = self.key_encoder.to(device)
        for param_q, param_k in zip(self.encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.queue = torch.randn(dim, queue_size, requires_grad=False).to(device)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, requires_grad=False).to(device)

    def forward(self, batch):
        query, key = batch[0][0].to(self.device), batch[0][1].to(self.device)
        # X: N, C, H, W
        query = self.encoder(query)
        query = nn.functional.normalize(query, dim=1)

        with torch.no_grad():
            self.update_key_()
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

        self.dequeue_and_enqueue_(key)
        loss = self.criterion(logits, labels)
        return loss

    @torch.no_grad()
    def dequeue_and_enqueue_(self, keys):
        batch_size = keys.size(0)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_ptr)
        # self.queue[:, ptr:ptr + batch_size].data = keys.T  <- error 복사가 이루어지지 않음
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_key_(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


class RotNet(Framework):
    def __init__(self, device, args):
        model = load_model(args.model, class_num=4)
        super().__init__(model, criterion=nn.CrossEntropyLoss(), device=device)
        dim_mlp = self.encoder.out.weight.shape[1]
        self.encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 4),
        )
        self.encoder = self.encoder.to(device)

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
        loss = self.criterion(output, label)
        return loss


class Simclr(Framework):
    def __init__(self, device, args, dim=128, tau=0.07):
        model = load_model(args.model, class_num=128)
        super().__init__(model, criterion=nn.CrossEntropyLoss(), device=device)
        self.tau = tau

        dim_mlp = self.encoder.out.weight.shape[1]
        self.encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim),
        )
        self.encoder = self.encoder.to(device)

    def forward(self, query, key):
        # query, key -> (batch, 3, 32, 32)
        query = self.encoder(query)
        query = nn.functional.normalize(query, dim=1)

        key = self.encoder(key)
        key = nn.functional.normalize(key, dim=1)

        # (2*batch, 128) query-key 쌍 생성
        logit = torch.cat([query, key], dim=0)
        # (2*batch,2*batch)
        logit = torch.mm(logit, logit.T) / self.tau
        logit = logit.fill_diagonal_(-1e9)
        # 자기꺼는 제외

        label = torch.cat([torch.arange(query.size(0), 2 * query.size(0)),
                          torch.arange(0, query.size(0))], dim=0)
        label = label.to(query.device)
        # (2*batch, batch)가 나옴
        loss = self.criterion(logit, label)
        return loss

    def save_model(self, output):
        torch.save(self.encoder, output)


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
