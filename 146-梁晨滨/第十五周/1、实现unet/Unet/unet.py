import torch
import torch.nn as nn
import torch.nn.functional as F


# 两层卷积块(conv + BN + Relu)
class Double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.conv2(x)



# 上采样
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Double_conv(in_channels, out_channels)

    def forward(self, x, x1):
        x_1 = self.up(x)
        x2 = torch.cat([x1, x_1], dim=1)
        out = self.conv(x2)
        return out


# 下采样
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.down(x)

# 总网络
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv_1 = Double_conv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.conv_2 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out_feature = self.conv_2(x)
        return out_feature

