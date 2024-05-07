import torch
import module.test
from tqdm import tqdm

def train(model,criterion,optimizer,epoch,lr,trainloader,testloader=None,print_step=1000,device=None):
    if device==None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for iter in range(epoch):
        total_loss=0    
        running_loss=0
        for idx, data in tqdm(enumerate(trainloader,start=0)):
            inputs, labels=data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            total_loss+=loss.item()
            if idx%print_step==print_step-1:
                print("epoch :",iter+1,", step : ",idx+1,", loss: ",running_loss/print_step)
                running_loss=0
        total_loss/=len(trainloader)
        if(testloader!=None):
            acc=module.test.test_model(model,testloader,device)
            print("Epoch :",iter,"end loss: ",total_loss," acc: ",acc)
        else:
            print("Epoch :",iter,"end loss: ",total_loss)
    return model