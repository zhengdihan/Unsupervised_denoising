import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x

class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)

        return x
    
class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x

class up_no_skip(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_no_skip, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, 
                        diffX // 2, diffX - diffX // 2), 'replicate')
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)

        return x

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.up1(self.x5, self.x4)
        self.x7 = self.up2(self.x6, self.x3)
        self.x8 = self.up3(self.x7, self.x2)
        self.x9 = self.up4(self.x8, self.x1)
        self.y = self.outc(self.x9)
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)

        return self.y
    
class UNet5(nn.Module):

    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet5, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.down5(self.x5)
        self.x7 = self.up1(self.x6, self.x5)
        self.x8 = self.up2(self.x7, self.x4)
        self.x9 = self.up3(self.x8, self.x3)
        self.x10 = self.up4(self.x9, self.x2)
        self.x11 = self.up5(self.x10, self.x1)
        self.y = self.outc(self.x11)
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)

        return self.y    

    
class UNet3(nn.Module):
    
    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.up1(self.x4, self.x3)
        self.x6 = self.up2(self.x5, self.x2)
        self.x7 = self.up3(self.x6, self.x1)
        self.y = self.outc(self.x7)
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)

        return self.y
    
class UNet2(nn.Module):
    
    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.up1(self.x3, self.x2)
        self.x5 = self.up2(self.x4, self.x1)
        self.y = self.outc(self.x5)
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)

        return self.y    

class UNet5_no_skip(nn.Module):

    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet5_no_skip, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)
        self.up1 = up_no_skip(1024, 1024)
        self.up2 = up_no_skip(1024, 512)
        self.up3 = up_no_skip(512, 256)
        self.up4 = up_no_skip(256, 128)
        self.up5 = up_no_skip(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        y = self.outc(x)
        if self.need_sigmoid:
            y = torch.sigmoid(y)

        return y

class UNet_no_skip(nn.Module):

    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet_no_skip, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_no_skip(512, 512)
        self.up2 = up_no_skip(512, 256)
        self.up3 = up_no_skip(256, 128)
        self.up4 = up_no_skip(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        y = self.outc(x)
        if self.need_sigmoid:
            y = torch.sigmoid(y)

        return y
    
class UNet3_no_skip(nn.Module):

    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet3_no_skip, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up_no_skip(256, 256)
        self.up2 = up_no_skip(256, 128)
        self.up3 = up_no_skip(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        y = self.outc(x)
        if self.need_sigmoid:
            y = torch.sigmoid(y)

        return y
    
class UNet2_no_skip(nn.Module):

    def __init__(self, n_channels, n_classes, need_sigmoid=False):
        super(UNet2_no_skip, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.up1 = up_no_skip(128, 128)
        self.up2 = up_no_skip(128, 64)
        self.outc = outconv(64, n_classes)
        self.need_sigmoid = need_sigmoid

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        y = self.outc(x)
        if self.need_sigmoid:
            y = torch.sigmoid(y)

        return y
