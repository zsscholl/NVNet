import torch
import torch.nn as nn
from .config import *
from .utils import *

class NVNet(nn.Module):

    class EncodeStep(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.Process = nn.Sequential(
                nn.Conv2d(in_channels=input_size, out_channels=output_size,
                          kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(output_size),
                nn.ReLU(),
                nn.Conv2d(in_channels=output_size, out_channels=output_size,
                          kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(output_size),
                nn.ReLU(),
            )
        def forward(self, x):
            return self.Process(x)

    class DecodeStep(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.Deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )
        def forward(self, x_last, x_skip):
            return self.Deconv(torch.cat([x_last, x_skip], 1))

    class EncodeBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = NVNet.EncodeStep(3, 8)
            self.enc2 = NVNet.EncodeStep(8, 16)
        def forward(self, x):
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            return [x2, x1]

    class DecodeBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.dec1 = NVNet.DecodeStep(32, 8)
            self.dec2 = NVNet.DecodeStep(16, 3)
            self.final = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        def forward(self, encoder_outputs):
            x4 = self.dec1(encoder_outputs[0], encoder_outputs[0])
            x5 = self.dec2(x4, encoder_outputs[1])
            x5 = self.final(x5)
            return x5

    def __init__(self):
        super().__init__()
        self.Encode = NVNet.EncodeBranch()
        self.DecodeX = NVNet.DecodeBranch()
        self.DecodeY = NVNet.DecodeBranch()
        # self.DecodeZ = NVNet.DecodeBranch()

    def forward(self, x):
        encoded = self.Encode(x)
        x_decoded = self.DecodeX(encoded)
        y_decoded = self.DecodeY(encoded)
        # z_decoded = self.DecodeZ(encoded)
        # print(f'output shape is {x_decoded.shape}')
        return torch.cat((x_decoded, y_decoded, torch.zeros_like(y_decoded)), dim=1)

# I've set the z output to 0 because I've had issues forward transforming a given magnetization into the correct stray
# field when it's not 0.

class SmartNet(nn.Module):
    class Encode(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
            )
        def forward(self, x):
            return self.encode(x)

    class Decode(nn.Module):
        def __init__(self):
            super().__init__()
            self.decode = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, padding=0, output_padding=0),
            )
        def forward(self, x):
            return self.decode(x)

    def __init__(self):
        super().__init__()
        self.encode = SmartNet.Encode()
        self.decode_x = SmartNet.Decode()
        self.decode_y = SmartNet.Decode()
        # self.decode_z = SmartNet.Decode()

    def forward(self, x):
        enc = self.encode(x)
        x = self.decode_x(enc)
        y = self.decode_y(enc)
        # z = self.decode_z(enc)
        return torch.cat((x, y, torch.zeros_like(y)), dim=1)