import torch
import torchvision
import torchvision.transforms as transforms


DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "stl10": torchvision.datasets.STL10,
}

basic_transform = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

TRANSFORMS = {
    "cifar10": transforms.Compose(basic_transform),
    "cifar100": transforms.Compose(basic_transform),
    "stl10": transforms.Compose(basic_transform),
}
CLASSNUM = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
}


def load_data(name="cifar10", root="./data", train=True, batch_size=16, shuffle=True):
    if name not in DATASETS:
        raise Exception("Unknown dataset name")
    dataset_class = DATASETS[name]
    transform = TRANSFORMS[name]
    class_num = CLASSNUM[name]

    dataset = dataset_class(root=root, train=train,
                            download=True, transform=transform)
    dataset.class_num = class_num
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
