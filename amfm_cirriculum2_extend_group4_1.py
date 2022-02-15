import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from attention.cbam import *

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(
                in_channels,
                2 * out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, feat_bn):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(num_features=feat_bn)
        self.cbam = CBAM(out_channels, 16)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.bn(x)
        x = self.conv(x)
        x_att = self.cbam(x)

        out = torch.max(x, x_att)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            mfm(in_channels=1, out_channels=32, kernel_size=[7, 3], padding=(1, 7), stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(32, 48, 3, 1, 1, 32),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm2d(num_features=48),
            group(48, 64, 3, 1, 1, 48),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(64, 32, 3, 1, 1, 64),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        self.features2 = nn.Sequential(
            group(32, 16, 3, 1, 1, 32),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.pre_mid = mfm(1152, 100, type=0)
        self.out_mid = nn.Linear(100, 3)

        self.features3 = nn.Sequential(
            group(32, 16, 3, 1, 1, 32),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(1152, 100, type=0)  
        self.fc_last = nn.Linear(100, 10)

    def forward(self, x):
        x = self.features(x)

        pre_pre_ = self.features2(x)
        pre_pre = pre_pre_.view(pre_pre_.size(0), -1)
        pre_pre = F.dropout(pre_pre, training=self.training)
        pre_mid = self.pre_mid(pre_pre)
        out_mid = self.out_mid(pre_mid)

        x = self.features3(x)
        x = x.view(x.size(0), -1)

        x = F.dropout(x, training=self.training)

        x = self.fc1(x)
        out_asc = self.fc_last(x)

        return out_mid, out_asc
