import torch
import torch.nn as nn
from ansatz.backend import utils

kernels = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
strides = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
pads = utils.dynamic_pad(kernels, strides, l=55)

class CleverNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernels[0], padding=pads[0], stride=strides[0]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernels[1], padding=pads[1], stride=strides[1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernels[2], padding=pads[2], stride=strides[2]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernels[3], padding=pads[3], stride=strides[3]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernels[4], padding=pads[4], stride=strides[4]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernels[5], padding=pads[5], stride=strides[5]),
            nn.LeakyReLU(),
        )
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kernels[6], padding=pads[6],
                               stride=strides[6]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=kernels[7], padding=pads[7], stride=strides[7]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=kernels[8], padding=pads[8], stride=strides[8]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=kernels[9], padding=pads[9], stride=strides[9]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=kernels[10], padding=pads[10], stride=strides[10]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=kernels[11], padding=pads[11], stride=strides[11]),
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kernels[6], padding=pads[6],
                               stride=strides[6]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=kernels[7], padding=pads[7], stride=strides[7]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=kernels[8], padding=pads[8], stride=strides[8]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=kernels[9], padding=pads[9], stride=strides[9]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=kernels[10], padding=pads[10], stride=strides[10]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=kernels[11], padding=pads[11], stride=strides[11]),
        )
    def forward(self, x):
        x = self.encode(x)
        x1 = self.decode1(x)
        x2 = self.decode2(x)
        return torch.cat((x1, x2, torch.zeros_like(x2)), 1)


