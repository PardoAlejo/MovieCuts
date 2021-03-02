import torch
from torch import nn
import torch.nn.functional as F
import resnet

class AVENet(nn.Module):

    def __init__(self, model_depth, n_classes, pool='avgpool'):
        super(AVENet, self).__init__()
        self.audnet = Resnet(model_depth, n_classes, pool='avgpool')

    def forward(self, audio):
        aud = self.audnet(audio)
        return aud


def Resnet(model_depth, n_classes, pool='avgpool'):

    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            num_classes=n_classes)
    elif model_depth == 18:
        model = resnet.resnet18(
            num_classes=n_classes,
            pool=pool)
    elif model_depth == 34:
        model = resnet.resnet34(
            num_classes=n_classes,
            pool=pool)
    elif model_depth == 50:
        model = resnet.resnet50(
            num_classes=n_classes,
            pool=pool)
    elif model_depth == 101:
        model = resnet.resnet101(
            num_classes=n_classes)
    elif model_depth == 152:
        model = resnet.resnet152(
            num_classes=n_classes)
    elif model_depth == 200:
        model = resnet.resnet200(
            num_classes=n_classes)
    return model 