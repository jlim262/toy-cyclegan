import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, stride, padding, norm=True, activation=True):
    sequence = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)]
    if(norm):
        sequence.append(nn.InstanceNorm2d(out_channels))
    if(activation):
        sequence.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*sequence)

def deconv(in_channels, out_channels, kernel_size, stride, padding, norm=True, activation=True):
    sequence = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)]
    if(norm):
        sequence.append(nn.InstanceNorm2d(out_channels))
    if(activation):
        sequence.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*sequence)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=False)
        self.conv2 = conv(out_channels, out_channels*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv(out_channels*2, out_channels*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv(out_channels*4, out_channels*8, kernel_size=4, stride=1, padding=1)
        self.conv5 = conv(out_channels*8, 1, kernel_size=4, stride=1, padding=1, activation=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x):
        pass

if __name__ == '__main__':
    print('test Discriminator')
    x = torch.randn(1, 1, 128, 128)
    d = Discriminator(1)
    output = d.forward(x)
