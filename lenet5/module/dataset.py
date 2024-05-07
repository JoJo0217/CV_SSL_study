import torch
import torchvision
import torchvision.transforms as transforms

def load_dataloader(root='./data',train=True,test=True,batch_size=16,shuffle=True):
    transform=transforms.Compose([
        transforms.ToTensor(),#tensor로 변경
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #평균과, 표준편차로 normalize 그냥 ToTensor만 하면 0~1로 값이 줄어들게 되는데 이를 평균화
    ])
    trainloader=None
    testloader=None
    if train:
        trainset=torchvision.datasets.CIFAR10(root=root,train=True,download=True,transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=shuffle)
    if test:
        testset=torchvision.datasets.CIFAR10(root=root,train=False,download=True,transform=transform)
        testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=shuffle)
    
    classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,testloader

