# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv -> BN -> ReLU) x2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """
    A simple UNet for depth completion.

    Input : [B, in_channels, H, W]  (default in_channels=5)
    Output: [B, 1, H, W]            (dense depth in meters, positive)
    """
    def __init__(self, in_channels: int = 5, base_channels: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)            # H, W
        self.enc2 = ConvBlock(base_channels, base_channels * 2)      # H/2, W/2
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)  # H/4, W/4
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)  # H/8, W/8

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder (upsample + concat skip + conv)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output head (1 channel depth)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

        # Keep depth positive
        self.out_act = nn.Softplus(beta=1.0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        y = self.out_conv(d1)
        y = self.out_act(y)
        return y
