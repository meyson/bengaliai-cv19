import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from efficientnet_pytorch import EfficientNet


class ResNet18(nn.Module):
    def __init__(self, pretrained=True, use_rgb=True):
        super(ResNet18, self).__init__()

        if pretrained:
            self.model = pretrainedmodels.models.resnet18(pretrained='imagenet')
        else:
            self.model = pretrainedmodels.models.resnet18(pretrained=False)

        # FIXME
        if not use_rgb:
            # surgery
            first_conv = nn.Conv2d(1, self.model.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            init.kaiming_uniform_(first_conv.weight)
            self.model.conv1 = first_conv

        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        N = x.shape[0]

        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(N, -1)
        s0 = self.l0(x)
        s1 = self.l1(x)
        s2 = self.l2(x)
        return s0, s1, s2


class SqueezeNet(nn.Module):
    def __init__(self, pretrained=True, use_rgb=True):
        super(SqueezeNet, self).__init__()

        if pretrained:
            self.model = pretrainedmodels.models.squeezenet1_1(pretrained='imagenet')
        else:
            self.model = pretrainedmodels.models.squeezenet1_1(pretrained=None)

        if not use_rgb:
            # surgery
            first_conv = nn.Conv2d(1, 64, kernel_size=3, stride=2)
            init.kaiming_uniform_(first_conv.weight)
            self.model.features[0] = first_conv

        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        N = x.shape[0]

        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(N, -1)
        s0 = self.l0(x)
        s1 = self.l1(x)
        s2 = self.l2(x)
        return s0, s1, s2


from efficientnet_pytorch import EfficientNet

class EfficientNetB3(nn.Module):
    def __init__(self, pretrained=True, use_rgb=True):
        super(EfficientNetB3, self).__init__()

        in_channels = 3 if use_rgb else 1

        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b3', in_channels=in_channels)
        else:
            pass

        self.bn = nn.BatchNorm2d(1536)
        # self.do = nn.Dropout(p=0.3)
        self.l0 = nn.Linear(1536, 168)
        self.l1 = nn.Linear(1536, 11)
        self.l2 = nn.Linear(1536, 7)

    def forward(self, x):
        N = x.shape[0]

        x = self.model.extract_features(x)
        x = self.bn(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(N, -1)
        # x = self.do(x)
        s0 = self.l0(x)
        s1 = self.l1(x)
        s2 = self.l2(x)
        return s0, s1, s2