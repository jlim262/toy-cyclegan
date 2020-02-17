import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride, padding, norm=True):
    sequence = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)]
    if(norm):
        sequence += [(nn.InstanceNorm2d(out_channels))]
    return nn.Sequential(*sequence)


def deconv(in_channels, out_channels, kernel_size, stride, padding, output_padding, norm=True):
    sequence = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True)]
    if(norm):
        sequence += [(nn.InstanceNorm2d(out_channels))]
    return nn.Sequential(*sequence)


def residual(dim, use_dropout, n_blocks):
    sequence = [ResidualBlock(dim, use_dropout)] * n_blocks
    return nn.Sequential(*sequence)


class ResidualBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.conv(dim, use_dropout)

    def conv(self, dim, use_dropout):
        sequence = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), 
            nn.InstanceNorm2d(dim),
            nn.ReLU(True)]

        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        sequence += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim)]

        return nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=False)
        self.conv2 = conv(out_channels, out_channels*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv(out_channels*2, out_channels*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv(out_channels*4, out_channels*8, kernel_size=4, stride=1, padding=1)
        self.conv5 = conv(out_channels*8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)      # C64
        out = F.leaky_relu(self.conv2(out), 0.2)    # C128
        out = F.leaky_relu(self.conv3(out), 0.2)    # C256
        out = F.leaky_relu(self.conv4(out), 0.2)    # C512
        out = self.conv5(out)
        return out


# Generator
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels=64, use_dropout=False):
        super(Generator, self).__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = conv(in_channels, out_channels, 7, 1, 0)               # conv
        self.conv2 = conv(out_channels, out_channels*2, 3, 2, 1)            # downsampling
        self.conv3 = conv(out_channels*2, out_channels*4, 3, 2, 1)          # downsampling
        self.residuals = residual(out_channels*4, use_dropout, n_blocks=9)
        self.deconv1 = deconv(out_channels*4, out_channels*2, 3, 2, 1, 1)   # upsampling
        self.deconv2 = deconv(out_channels*2, out_channels, 3, 2, 1, 1)     # upsampling
        self.conv4 = conv(out_channels, 3, 7, 1, 0, norm=False)             # conv

    def forward(self, x):
        out = self.pad(x)
        out = F.relu(self.conv1(out), True)     # c7s1-64
        out = F.relu(self.conv2(out), True)     # d128
        out = F.relu(self.conv3(out), True)     # d256
        out = self.residuals(out)               # R256 * 9
        out = F.relu(self.deconv1(out), True)   # u128
        out = F.relu(self.deconv2(out), True)   # u64
        out = self.pad(out)
        out = torch.tanh(self.conv4(out))       # c7s1-3
        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)

    print('test Discriminator')
    d = Discriminator(3)
    d.forward(x)

    print('test Generator')
    g = Generator(3)
    g.forward(x)
