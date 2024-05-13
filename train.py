import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
from eval import eval_model


def train(
        model, criterion, optimizer, 
        epoch, trainloader, testloader=None, 
        device=None, logging_step=None,
        log_dir=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
    logger=Logger(log_dir)
    global_step = 0
    log_running_loss = 0
    for iter in range(epoch):
        total_loss = 0    
        
        for idx, data in tqdm(enumerate(trainloader, start=0)):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            log_running_loss += loss.item()
            total_loss += loss.item()
            
            if logging_step is not None and global_step!=0 and (global_step%logging_step) == 0:
                logger.log(global_step, loss=log_running_loss/logging_step)
                log_running_loss = 0
            global_step += 1

        total_loss /= len(trainloader)
        
        if(testloader is not None):
            acc = eval_model(model, testloader, device)
            print(f"Epoch : {iter}, end loss: {total_loss}, acc: {acc}")
        else:
            print(f"Epoch : {iter}, end loss: {total_loss}")
    return model