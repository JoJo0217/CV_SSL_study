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


class MoCo(torch.nn.Module):
    def __init__(self, device, args, dim=128, queue_size=65536, m=0.999, tau=0.07):
        super().__init__()
        self.dim = dim
        self.m = m
        self.tau = tau
        self.queue_size = queue_size
        self.query_encoder = load_model(args.model, class_num=dim)
        self.query_encoder = self.query_encoder.to(device)
        self.key_encoder = load_model(args.model, class_num=dim)
        self.key_encoder = self.key_encoder.to(device)

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(32, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # in the paper moco use encoder with average pool layer as output
        dim_mlp = self.query_encoder.out.weight.shape[1]

        self.key_encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim),
        )
        self.query_encoder.out = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim),
        )
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.queue = torch.randn(dim, queue_size).to(device)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        # X: N, C, H, W
        query = self.augment(x)
        key = self.augment(x)
        query = self.query_encoder(query)
        key = self.key_encoder(key)
        # (N, 128)
        l_pos = torch.bmm(query.view(query.size(0), 1, -1),
                          key.view(key.size(0), -1, 1)).squeeze(-1)  # (N,1,128) (N,128,1) -> (N,1,1) -> (N,1)
        # (N,1)
        l_neg = torch.mm(query, self.queue)
        # (N,128) (128,queue_size) -> (N,queue_size)

        # (N,1)+(N,queue_size) -> (N,queue_size+1)
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.tau
        labels = torch.zeros(logits.size(0)).long().to(
            logits.device)  # 0번이 positive니까

        self.dequeue_and_enqueue(key)
        return logits, labels

    def dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)

        assert self.queue_size % batch_size == 0

        ptr = int(self.queue_ptr)
        new_queue = self.queue.clone()
        new_queue[:, ptr:ptr + batch_size] = keys.T
        self.queue = new_queue
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def update_key(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def save_model(self, output):
        torch.save(self.query_encoder, output)


TRAINERS = {
    "moco": MoCo,
}


def load_trainer(args, device):
    if args.pretrain not in TRAINERS:
        raise Exception("Unknown pretrain method")
    else:
        return TRAINERS[args.pretrain](device, args)
