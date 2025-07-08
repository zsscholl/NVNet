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
        # gap = x.shape[-1] - DATA_CONFIG['SHAPE']
        # slice_start, slice_end = gap//2, gap//2+DATA_CONFIG['SHAPE']

        encoded = self.Encode(x)
        x_decoded = self.DecodeX(encoded) #[:, :, slice_start:slice_end, slice_start:slice_end]
        y_decoded = self.DecodeY(encoded) #[:, :, slice_start:slice_end, slice_start:slice_end]
        # z_decoded = self.DecodeZ(encoded) #[:, :, slice_start:slice_end, slice_start:slice_end]
        # print(f'output shape is {x_decoded.shape}')
        return torch.cat((x_decoded, y_decoded, torch.zeros_like(y_decoded)), dim=1)


