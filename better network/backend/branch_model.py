import torch
import torch.nn as nn
from .config import *
from .utils import *

class UBranch(nn.Module):
    class Encode(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_process = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2)
            self.mask_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2)
            self.encoder = nn.Sequential(
                nn.BatchNorm2d(num_features=8),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=16),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU()
            )
        def forward(self, x, mask):
            x = self.pre_process(x)
            mask = self.mask_conv(mask)
            roi = x*mask
            return self.encoder(roi)

    class Decode(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2,
                                   output_padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2,
                                   output_padding=1),
                nn.BatchNorm2d(num_features=32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2,
                                   output_padding=1),
                nn.BatchNorm2d(num_features=16),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2,
                                   output_padding=1),
                nn.BatchNorm2d(num_features=8),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        def forward(self, x):
            x = self.decoder(x)
            return x

    def __init__(self):
        super().__init__()
        self.enc = UBranch.Encode()
        self.dec = UBranch.Decode()

    def forward(self, x, mask):
        x = self.enc(x, mask)
        x = self.dec(x)
        return x