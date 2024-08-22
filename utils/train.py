import torch
import torch.nn as nn
from tqdm import tqdm
from utils.logger import Logger
from utils.evaluate import eval_model, eval_pretrain_model

AFTER_EPOCH_SCHEDULER = [
    "reduce_on_plateau",
    "multi_step",
    "cos_annealing"
]


def train(
        model, criterion, optimizer,
        epoch, trainloader, testloader=None,
        device=None, logging_step=None,
        logger=None, scheduler=None, scheduer_type=None,
        grad_clip=None, is_pretrain=None, pretrainloader=None):
    if logger is None:
        logger = Logger(None)

    global_step = 0
    log_running_loss = 0

    if is_pretrain is not None:
        test_trainloader = trainloader
        trainloader = pretrainloader

    for iter in range(epoch):
        total_loss = 0
        model.train()
        for idx, data in tqdm(enumerate(trainloader, start=0)):

            optimizer.zero_grad()
            loss = model(data)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            log_running_loss += loss.item()
            total_loss += loss.item()

            if logging_step is not None and global_step != 0 and (global_step % logging_step) == 0:
                logger.log(global_step, loss=log_running_loss /
                           logging_step, lr=optimizer.param_groups[0]["lr"])
                log_running_loss = 0
            global_step += 1

        total_loss /= len(trainloader)
        if scheduler is not None and scheduer_type in AFTER_EPOCH_SCHEDULER:
            if scheduer_type == "reduce_on_plateau":
                scheduler.step(total_loss)
            else:
                scheduler.step()

        if (testloader is not None):
            model.eval()
            if is_pretrain is not None:
                acc = eval_pretrain_model(
                    model, test_trainloader, testloader, device)
            else:
                acc = eval_model(model, testloader, device)
            logger.log(global_step, epoch=iter, loss=total_loss,
                       acc=acc, lr=optimizer.param_groups[0]["lr"])
        else:
            logger.log(global_step, epoch=iter, loss=total_loss,
                       lr=optimizer.param_groups[0]["lr"])
    return model


class Framework(nn.Module):
    def __init__(self, encoder, criterion=nn.CrossEntropyLoss(), device=None):
        super().__init__()
        self.encoder = encoder
        self.criterion = criterion
        self.device = device

    def forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self.encoder(x)
        return self.criterion(pred, y)

    def evaluate(self, x):
        x = x.to(self.device)
        return self.encoder(x)

    def extract_features(self, x):
        x = x.to(self.device)
        return self.encoder.extract_features(x)

    def save(self, path):
        torch.save(self.encoder, path)
