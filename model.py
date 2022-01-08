import torch
from torch import nn


# class UNet(nn.Module):
#     def __init__(self, in_chan, outchan):
#         super(UNet, self).__init__()
#         self.conv_group_down_1 = nn.Sequential(
#             nn.Conv2d(in_chan, 64, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.LeakyReLU(0.2)
#         )  # (101,101) -> (50,50)
#         self.conv_group_down_2 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.LeakyReLU(0.2)
#         )  # (50,50) -> (25,25)
#         self.conv_group_down_3 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.LeakyReLU(0.2)
#         )  # (25,25) -> (12,12)
#         self.conv_group_down_4 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.LeakyReLU(0.2),
#         )  # (12,12) -> (6,6)
#         self.conv_group_bottom = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(512, 1024, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1024, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1024, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             # nn.UpsamplingNearest2d(scale_factor=2)
#         )  # (6,6)
#         self.conv_group_up_1 = nn.Sequential(
#             nn.Conv2d(1024+512, 512, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv_group_up_2 = nn.Sequential(
#             nn.Conv2d(512+256, 256, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv_group_up_3 = nn.Sequential(
#             nn.Conv2d(256+128, 128, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.LeakyReLU(0.2),
#         )
#         self.conv_group_up_4 = nn.Sequential(
#             nn.Conv2d(128+64, 64, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 1, 3, padding=1)
#         )
#         self.upsample1 = nn.UpsamplingNearest2d(size=(12, 12))
#         self.upsample2 = nn.UpsamplingNearest2d(size=(25, 25))
#         self.upsample3 = nn.UpsamplingNearest2d(size=(50, 50))
#         self.upsample4 = nn.UpsamplingNearest2d(size=(101, 101))
#         return

#     def forward(self, x):

#         x1 = self.conv_group_down_1(x)
#         x2 = self.conv_group_down_2(x1)
#         x3 = self.conv_group_down_3(x2)
#         x4 = self.conv_group_down_4(x3)
#         x5 = self.conv_group_bottom(x4)
#         # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
#         x6 = self.conv_group_up_1(torch.cat([self.upsample1(x5), x4], dim=1))
#         # print(self.upsample(x6).shape, x3.shape)
#         x7 = self.conv_group_up_2(torch.cat([self.upsample2(x6), x3], dim=1))
#         x8 = self.conv_group_up_3(torch.cat([self.upsample3(x7), x2], dim=1))
#         x9 = self.conv_group_up_4(torch.cat([self.upsample4(x8), x1], dim=1))
#         return x9

#     

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    def train_step(self, x, t, loss_fn, optimizer, metric_fn):
        output = self.forward(x)
        loss = loss_fn(output, t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_iou = metric_fn(output, t).mean()
        return loss.item(), mean_iou.item()
    
    
if(__name__ == '__main__'):
    model=UNet(1,1)
    x=torch.randn((2,1,101,101))
    print(model(x).shape)