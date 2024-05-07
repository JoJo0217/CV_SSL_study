import torch
from torch import nn
import torch.optim as optim
from module.model import Lenet_5
from module.dataset import load_dataloader
from module.train import train
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#hyper param
epoch=10
lr=0.001
batch_size=64
print_step=100

#load dataset
trainloader,testloader=load_dataloader(batch_size=batch_size,train=True, test=True)


model=Lenet_5()
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#train
print("start training")
model=train(model,criterion,optimizer,epoch,lr,trainloader,testloader,print_step)
print("finish training")

print("saving...")
torch.save(model,'model.pth')
print("save success")