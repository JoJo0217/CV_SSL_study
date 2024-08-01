import torch
import torchvision
import torchvision.transforms as transforms

train_transform = [
    transforms.RandomResizedCrop((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

basic_transform = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

pretrain_transform = [
    transforms.RandomResizedCrop(32, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "stl10": torchvision.datasets.STL10,
}


TRAIN_TRANSFORMS = {
    "cifar10": transforms.Compose(train_transform),
    "cifar100": transforms.Compose(train_transform),
    "stl10": transforms.Compose(train_transform),
}

TEST_TRANSFORMS = {
    "cifar10": transforms.Compose(basic_transform),
    "cifar100": transforms.Compose(basic_transform),
    "stl10": transforms.Compose(basic_transform),
}
CLASSNUM = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
}


class PretrainSplitter():
    def __init__(self, transform):
        self.augment = transforms.Compose(transform)

    def __call__(self, x):
        q = self.augment(x)
        k = self.augment(x)
        return [q, k]


def load_data(name="cifar10", root="./data", train=True, batch_size=16, shuffle=True, not_transform=False, drop_last=False):
    if name not in DATASETS:
        raise Exception("Unknown dataset name")
    dataset_class = DATASETS[name]
    if train is True:
        transform = TRAIN_TRANSFORMS[name]
    else:
        transform = TEST_TRANSFORMS[name]

    class_num = CLASSNUM[name]
    if not_transform and train:
        # transform = None
        transform = PretrainSplitter(pretrain_transform)
    dataset = dataset_class(root=root, train=train,
                            download=True, transform=transform)
    dataset.class_num = class_num
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
