import torch

from dipNV.backend.packages import *

class NVNet(nn.Module):
    class EncoderBlock(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.activ = nn.LeakyReLU(negative_slope=0.2)
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.bn = nn.BatchNorm2d(out_channel)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.activ(x)
            x = self.conv2(x)
            x = self.activ(x)
            x = self.bn(x)
            x = self.pool(x)
            return x

    class DecoderBlock(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.activ = nn.LeakyReLU(negative_slope=0.2)
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
            self.bn = nn.BatchNorm2d(out_channel)

        def forward(self, x):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.conv1(x)
            x = self.activ(x)
            x = self.conv2(x)
            x = self.activ(x)
            x = self.bn(x)
            return x

    class EncoderStage(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.depth = depth
            self.encode = nn.Sequential(
                NVNet.EncoderBlock(1, 8*self.depth),
                NVNet.EncoderBlock(8*self.depth, 16*self.depth),
                # NVNet.EncoderBlock(16*self.depth, 32*self.depth),
                # NVNet.EncoderBlock(32*self.depth, 64*self.depth),
                # NVNet.EncoderBlock(64*self.depth, 128*self.depth)
            )

        def forward(self, x):
            return self.encode(x)

    class DecoderStage(nn.Module):
        def __init__(self, depth, DO_RELU):
            super().__init__()
            self.depth = depth
            self.decode = nn.Sequential(
                # NVNet.DecoderBlock(128*self.depth, 64*self.depth),
                # NVNet.DecoderBlock(64*self.depth, 32*self.depth),
                # NVNet.DecoderBlock(32*self.depth, 16*self.depth),
                NVNet.DecoderBlock(16*self.depth, 8*self.depth),
                NVNet.DecoderBlock(8*self.depth, 1)
            )
            if DO_RELU is True:
                self.final = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
                    nn.ReLU()
                )
            else:
                self.final = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                )

        def forward(self, x):
            return self.final(self.decode(x))

    def __init__(self, depth, DO_RELU):
        super().__init__()
        self.encoder = NVNet.EncoderStage(depth)
        self.decoder_x = NVNet.DecoderStage(depth, DO_RELU)
        self.decoder_y = NVNet.DecoderStage(depth, DO_RELU)
        self.decoder_z = NVNet.DecoderStage(depth, DO_RELU)

    def forward(self, x):
        enc = self.encoder(x)
        dec_x = self.decoder_x(enc)
        dec_y = self.decoder_y(enc)
        dec_z = self.decoder_z(enc)
        return torch.cat([dec_x, dec_y, dec_z], 1)