import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()
        self.channel = channel
        self.conv = nn.Conv2d(channel, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        score = self.conv(x)
        x = x * torch.sigmoid(score)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.channel = channel
        self.conv = nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(channel, channel//ratio, 1, 1, 0, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(channel//ratio, channel, 1, 1, 0, bias=False))

    def forward(self, x):
        score = self.conv(x)
        x = x * torch.sigmoid(score)
        return x

class CBAM(nn.Module):
    def __init__(self, channel, ratio=8):
        super(CBAM, self).__init__()
        self.channel = channel
        self.conv = nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(channel, channel//ratio, 1, 1, 0, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(channel//ratio, channel, 1, 1, 0, bias=False))
        self.conv2 = nn.Conv2d(channel, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        shortcut = x
        score = self.conv(x)
        x = x * torch.sigmoid(score)
        score = self.conv2(x)
        x = x * torch.sigmoid(score)
        x = shortcut + x
        return x


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
