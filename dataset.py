import torch
import torchvision
import torchvision.transforms as transforms


def load_data(
        name="cifar10", root="./data", 
        train=True, batch_size=16, 
        shuffle=True):
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ])
        dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True, 
            transform=transform
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
            )
        return dataloader
    
    elif name == "cifar100":
        pass
    
    elif name == "stl10":
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ])
        split = "train" if train else "test"
        dataset = torchvision.datasets.STL10(
            root=root, 
            split=split, 
            download=True, 
            transform=transform
            )
        dataloader=torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
            )
        return dataloader
    else:
        raise Exception("Unknown dataset name")