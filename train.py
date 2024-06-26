import torch
from tqdm import tqdm
from logger import Logger
from evaluate import eval_model

AFTER_EPOCH_SCHEDULER = [
    "reduce_on_plateau",
    "multi_step",
    "cos_annealing"
]


def train(
        model, criterion, optimizer,
        epoch, trainloader, testloader=None,
        device=None, logging_step=None,
        logger=None, scheduler=None, scheduer_type=None):
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
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
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
            acc = eval_model(model, testloader, device)
            logger.log(global_step, epoch=iter, loss=total_loss,
                       acc=acc, lr=optimizer.param_groups[0]["lr"])
        else:
            logger.log(global_step, epoch=iter, loss=total_loss,
                       lr=optimizer.param_groups[0]["lr"])
    return model
