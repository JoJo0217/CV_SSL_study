import inspect
from torch import nn
import torch.optim as optim

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}

CRITERIONS = {
    "cross_entropy": nn.CrossEntropyLoss
}

SCHEDULERS = {
    "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau
    "multi_step": optim.lr_scheduler.MultiStepLR
}


def load_optimizer(name, model, **kwargs):
    if name not in OPTIMIZERS:
        raise Exception("Unknown optimizer name")
    optimizer = OPTIMIZERS[name]
    valid_args = inspect.getfullargspec(optimizer).args

    # Check if any of the kwargs are not valid for the optimizer
    valid_kwargs = {k: v for k, v in kwargs.items(
    ) if k in valid_args and v is not None}
    return optimizer(model.parameters(), **valid_kwargs)


def load_criterion(name):
    if name not in CRITERIONS:
        raise Exception("Unknown criterion name")
    return CRITERIONS[name]()


def load_scheduler(name, optimizer):
    if name == None:
        return None
    if name == "reduce_on_plateau":
        return SCHEDULERS[name](optimizer, verbose=True)
    if name == "multi_step":
        return SCHEDULERS[name](optimizer, milestones=[100, 150], gamma=0.1)
    if name not in SCHEDULERS:
        raise Exception("Unknown scheduler name")
