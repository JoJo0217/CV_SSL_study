from torch import nn
import torch.optim as optim


def load_optimizer(name, model, **kwargs):
    if name == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    else:
        raise Exception("Unknown optimizer name")

    
def load_criterion(name):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise Exception("Unknown criterion name")