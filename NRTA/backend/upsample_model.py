import torch

from NRTA.backend.packages import *
from NRTA.backend.utils import *

class NVNet(torch.nn.Module):

    class Encoder(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.activ = nn.LeakyReLU(negative_slope=0.2)
            self.Conv_3_8 = nn.Conv2d(in_channels=3, out_channels=8*depth, kernel_size=5, stride=2, padding=2)
            self.BN_8 = nn.InstanceNorm2d(8*depth)
            self.Conv_8_16 = nn.Conv2d(in_channels=8 * depth, out_channels=16 * depth, kernel_size=5,
                                       stride=2, padding=2)
            self.BN_16 = nn.InstanceNorm2d(16 * depth)
            self.Conv_16_32 = nn.Conv2d(in_channels=16 * depth, out_channels=32 * depth, kernel_size=5,
                                        stride=2, padding=2)
            self.BN_32 = nn.InstanceNorm2d(32 * depth)
            self.Conv_32_64 = nn.Conv2d(in_channels=32 * depth, out_channels=64 * depth, kernel_size=5,
                                        stride=2, padding=2)
            self.BN_64 = nn.InstanceNorm2d(64 * depth)
            self.Conv_64_128 = nn.Conv2d(in_channels=64 * depth, out_channels=128 * depth, kernel_size=5,
                                         stride=2, padding=2)
            self.BN_128 = nn.InstanceNorm2d(128 * depth)

        def forward(self, x):
            enc_8 = self.BN_8(self.activ(self.Conv_3_8(x)))
            enc_16 = self.BN_16(self.activ(self.Conv_8_16(enc_8)))
            enc_32 = self.BN_32(self.activ(self.Conv_16_32(enc_16)))
            enc_64 = self.BN_64(self.activ(self.Conv_32_64(enc_32)))
            enc_128 = self.BN_128(self.activ(self.Conv_64_128(enc_64)))
            return enc_8, enc_16, enc_32, enc_64, enc_128

    class Decoder(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.activ = nn.LeakyReLU(negative_slope=0.2)
            self.Deconv_128_64 = nn.Conv2d(in_channels=128 * depth, out_channels=64 * depth,
                                           kernel_size=5, stride=1, padding=2)
            self.BN_64 = nn.InstanceNorm2d(64 * depth)
            self.Deconv_64_32 = nn.Conv2d(in_channels=2*64 * depth, out_channels=32 * depth,
                                                   kernel_size=5, stride=1, padding=2)
            self.BN_32 = nn.InstanceNorm2d(32 * depth)
            self.Deconv_32_16 = nn.Conv2d(in_channels=2*32 * depth, out_channels=16 * depth,
                                                   kernel_size=5, stride=1, padding=2)
            self.BN_16 = nn.InstanceNorm2d(16 * depth)
            self.Deconv_16_8 = nn.Conv2d(in_channels=2*16 * depth, out_channels=8 * depth,
                                                  kernel_size=5, stride=1, padding=2)
            self.BN_8 = nn.InstanceNorm2d(8 * depth)
            self.Deconv_8_1 = nn.Conv2d(in_channels=2*8 * depth, out_channels=1,
                                                 kernel_size=5, stride=1, padding=2)
        def forward(self, enc_8, enc_16, enc_32, enc_64, enc_128):
            dec_64 = self.BN_64(self.activ(self.Deconv_128_64(nn.functional.interpolate(enc_128, scale_factor=2))))
            dec_32 = self.BN_32(self.activ(self.Deconv_64_32(nn.functional.interpolate(torch.cat([enc_64, dec_64], dim=1), scale_factor=2))))
            dec_16 = self.BN_16(self.activ(self.Deconv_32_16(nn.functional.interpolate(torch.cat([enc_32, dec_32], dim=1), scale_factor=2))))
            dec_8 = self.BN_8(self.activ(self.Deconv_16_8(nn.functional.interpolate(torch.cat([enc_16, dec_16], dim=1), scale_factor=2))))
            dec_1 = self.Deconv_8_1(nn.functional.interpolate(torch.cat([enc_8, dec_8], dim=1), scale_factor=2))
            return dec_1

    def __init__(self, data_dict):
        super().__init__()
        depth = data_dict.CONFIG['DEPTH']
        self.amp = data_dict.CONFIG['M_AMP']
        self.mask = data_dict.mask
        self.encoder = NVNet.Encoder(depth)
        self.decoder_x = NVNet.Decoder(depth)
        self.decoder_y = NVNet.Decoder(depth)
        # self.decoder_z = NVNet.Decoder(depth)

    def forward(self, x):
        encoded = self.encoder(x)
        dec_x = self.decoder_x(*encoded)
        dec_y = self.decoder_y(*encoded)
        m_vec = torch.cat([dec_x, dec_y, torch.zeros_like(dec_x)], dim=1)
        return (self.amp*m_vec)*self.mask


