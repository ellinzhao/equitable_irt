import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        norm = 'instancenorm'
        act = 'relu'

        norm_fns = {'batchnorm': nn.BatchNorm2d, 'instancenorm': nn.InstanceNorm2d}
        act_fns = {'relu': nn.ReLU, 'gelu': nn.GELU, 'elu': nn.ELU, 
                   'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU}

        norm_fn = norm_fns[norm]
        act_fn = act_fns[act]

        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_fn(mid_channels),
            act_fn(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_fn(out_channels),
            act_fn(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

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
    def __init__(self, in_ch=1, out_ch=2, ch_mult=4):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.chs = [ch_mult * i for i in [1, 2, 4, 8, 16]]

        assert len(self.chs) == 5
        factor = 2
        ch1, ch2, ch3, ch4, ch5 = self.chs

        self.inc = (DoubleConv(in_ch, ch1))
        self.down1 = (Down(ch1, ch2))
        self.down2 = (Down(ch2, ch3))
        self.down3 = (Down(ch3, ch4))
        self.down4 = (Down(ch4, ch5 // factor))
        self.up1 = (Up(ch5, ch4 // factor))
        self.up2 = (Up(ch4, ch3 // factor))
        self.up3 = (Up(ch3, ch2 // factor))
        self.up4 = (Up(ch2, ch1))
        self.outc = (OutConv(ch1, out_ch))

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
        x = self.outc(x)
        return x
