from models import lenet
from models import resnet


MODELS = {
    "lenet5": lenet.Lenet5,
    "resnet18": resnet.Resnet18,
    "resnet18bottleneck": resnet.ResBottleNecknet18,
    "resnet50bottleneck": resnet.ResBottleNecknet50,
    "resnet18preact": resnet.PreActResNet,
    "resnext": resnet.ResNext,
}


def load_model(name, class_num=10):
    if name not in MODELS:
        raise Exception("Unknown model name")
    return MODELS[name](class_num=class_num)
