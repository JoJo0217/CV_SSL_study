import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import transformer
from models import resnet
from models import lenet
from models import mixer


MODELS = {
    "lenet5": lenet.Lenet5,
    "resnet18": resnet.Resnet18,
    "resnet18bottleneck": resnet.ResBottleNecknet18,
    "resnet50bottleneck": resnet.ResBottleNecknet50,
    "resnet18preact": resnet.PreActResNet,
    "resnext": resnet.ResNext,
    "vit": transformer.ViT,
    "convnext": resnet.ConvNeXt,
    "mlpmixer": mixer.MLPMixer,
    "convmixer": mixer.ConvMixer,
}


def load_model(name, class_num=10):
    if name not in MODELS:
        raise Exception("Unknown model name")
    return MODELS[name](class_num=class_num)
