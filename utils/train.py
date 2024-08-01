import torch
from tqdm import tqdm
from utils.logger import Logger
from utils.evaluate import eval_model, eval_pretrain_model

AFTER_EPOCH_SCHEDULER = [
    "reduce_on_plateau",
    "multi_step",
    "cos_annealing"
]

TWO_INPUT_PRETRAIN = [
    "moco"
]


def train(
        model, criterion, optimizer,
        epoch, trainloader, testloader=None,
        device=None, logging_step=None,
        logger=None, scheduler=None, scheduer_type=None,
        grad_clip=None, is_pretrain=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    if logger is None:
        logger = Logger(None)

    global_step = 0
    log_running_loss = 0
    for iter in range(epoch):
        total_loss = 0

        for idx, data in tqdm(enumerate(trainloader, start=0)):
            labels = data[1].to(device)

            optimizer.zero_grad()

            if is_pretrain in TWO_INPUT_PRETRAIN:
                inputs = [d.to(device) for d in data[0]]
                outputs, labels = model(inputs[0], inputs[1])
            else:
                inputs = data[0].to(device)
                outputs = model(inputs)
            loss = criterion(outputs, labels)
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
            if is_pretrain is not None:
                acc = eval_pretrain_model(
                    model, trainloader, testloader, device, is_pretrain)
            else:
                acc = eval_model(model, testloader, device)
            logger.log(global_step, epoch=iter, loss=total_loss,
                       acc=acc, lr=optimizer.param_groups[0]["lr"])
        else:
            logger.log(global_step, epoch=iter, loss=total_loss,
                       lr=optimizer.param_groups[0]["lr"])
    return model
