import torch
import torch.nn as nn
from betternet.debugging import *


class NV_Net(nn.Module):

    class PreProcess(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.depth = parent.depth
            self.preprocess = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8*self.depth, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(8*self.depth),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=8*self.depth, out_channels=8*self.depth, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(8*self.depth),
                nn.LeakyReLU()
            )
        def forward(self, x):
            return self.preprocess(x)

    class PostProcess(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.depth = parent.depth
            self.postprocess = nn.Sequential(
                nn.ConvTranspose2d(in_channels=8*self.depth, out_channels=1, kernel_size=5, stride=1, padding=2, output_padding=0),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        def forward(self, x):
            return self.postprocess(x)

    class Encoder(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.depth = parent.depth
            self.activ = parent.activ
            self.IN_16 = parent.IN_16
            self.IN_32 = parent.IN_32
            self.IN_64 = parent.IN_64
            self.IN_128 = parent.IN_128
            self.Conv_8_16 = nn.Conv2d(in_channels=8*self.depth, out_channels=16*self.depth, kernel_size=5, stride=2, padding=2)
            self.Conv_16_32 = nn.Conv2d(in_channels=16*self.depth, out_channels=32*self.depth, kernel_size=5, stride=2, padding=2)
            self.Conv_32_64 = nn.Conv2d(in_channels=32*self.depth, out_channels=64*self.depth, kernel_size=5, stride=2, padding=2)
            self.Conv_64_128 = nn.Conv2d(in_channels=64*self.depth, out_channels=128*self.depth, kernel_size=5, stride=2, padding=2)
        def forward(self, x):
            enc_8 = x
            enc_16 = self.activ(self.IN_16(self.Conv_8_16(x)))
            enc_32 = self.activ(self.IN_32(self.Conv_16_32(enc_16)))
            enc_64 = self.activ(self.IN_64(self.Conv_32_64(enc_32)))
            enc_128 = self.activ(self.IN_128(self.Conv_64_128(enc_64)))
            return enc_8, enc_16, enc_32, enc_64, enc_128

    class Decoder(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.depth = parent.depth
            self.activ = parent.activ
            self.IN_8 = parent.IN_8
            self.IN_16 = parent.IN_16
            self.IN_32 = parent.IN_32
            self.IN_64 = parent.IN_64
            self.IN_128 = parent.IN_128
            self.Deconv_128_64 = nn.ConvTranspose2d(in_channels=128*self.depth, out_channels=64*self.depth,
                                                    kernel_size=5, stride=2, padding=2, output_padding=1)
            self.Deconv_64_32 = nn.ConvTranspose2d(in_channels=(64+64)*self.depth, out_channels=32*self.depth,
                                                   kernel_size=5, stride=2, padding=2, output_padding=1)
            self.Deconv_32_16 = nn.ConvTranspose2d(in_channels=(32+32)*self.depth, out_channels=16*self.depth,
                                                   kernel_size=5, stride=2, padding=2, output_padding=1)
            self.Deconv_16_8 = nn.ConvTranspose2d(in_channels=(16+16)*self.depth, out_channels=8*self.depth,
                                                  kernel_size=5, stride=2, padding=2, output_padding=1)
        def forward(self, enc_8, enc_16, enc_32, enc_64, enc_128):
            dec_64 = self.activ(self.IN_64(self.Deconv_128_64(enc_128)))
            dec_32 = self.activ(self.IN_32(self.Deconv_64_32(torch.cat([enc_64, dec_64], dim=1))))
            dec_16 = self.activ(self.IN_16(self.Deconv_32_16(torch.cat([enc_32, dec_32], dim=1))))
            dec_8 = self.activ(self.IN_8(self.Deconv_16_8(torch.cat([enc_16, dec_16], dim=1))))
            return dec_8

    def __init__(self, depth, activ):
        super().__init__()
        self.depth = depth
        self.activ = activ
        self.IN_8 = nn.InstanceNorm2d(8*self.depth)
        self.IN_16 = nn.InstanceNorm2d(16*self.depth)
        self.IN_32 = nn.InstanceNorm2d(32*self.depth)
        self.IN_64 = nn.InstanceNorm2d(64*self.depth)
        self.IN_128 = nn.InstanceNorm2d(128*self.depth)
        self.preprocess = NV_Net.PreProcess(self)
        self.encode = NV_Net.Encoder(self)
        self.decode = NV_Net.Decoder(self)
        self.postprocess = NV_Net.PostProcess(self)

    def forward(self, x):
        input = self.preprocess(x)
        encoded = self.encode(input)
        decoded = self.decode(*encoded)
        output = self.postprocess(decoded)*CLOVERMASK
        return output
