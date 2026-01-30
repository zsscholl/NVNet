import torch

from NVNet.backend.packages import *
from NVNet.backend.config import *
from NVNet.backend.utils import *
from NVNet.backend.fourier_operations import *
from NVNet.backend.masking import *

# landau_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_state.npy')[:, 7, :, :]
# LANDAU_DATA = torch.from_numpy(landau_raw).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
# rgb = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_rgb.png')
# grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)/256
#
# XY_MASK = toTorch(grayscale)
# XY_MASK = tv.transforms.functional.gaussian_blur(XY_MASK, (15, 15)).to(device=REC_CONFIG['DEVICE'])
# Z_MASK = torch.where(LANDAU_DATA[0, 2].abs() <= 100000, torch.zeros_like(XY_MASK), torch.ones_like(XY_MASK))
# Z_MASK = tv.transforms.functional.gaussian_blur(Z_MASK, (23, 23)).to(device=REC_CONFIG['DEVICE'])
# MASK = torch.cat((XY_MASK, XY_MASK, Z_MASK), dim=1)

# skyrmag = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\skyrmion_mag.npy')[0:3, 7, :, :]
# skyrmag = torch.from_numpy(skyrmag).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
# skyrmask = torch.where(skyrmag[0,0].abs()+skyrmag[0,1].abs() <= 100, torch.zeros_like(skyrmag[0]), torch.ones_like(skyrmag[0])).unsqueeze(0)
# DEFINING THE MODEL

class NV_Net(nn.Module):

    class PreProcess(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.depth = parent.depth
            self.preprocess = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8*self.depth, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(8*self.depth),
                nn.ReLU(),
                nn.Conv2d(in_channels=8*self.depth, out_channels=8*self.depth, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(8*self.depth),
                nn.ReLU()
            )
        def forward(self, x):
            return self.preprocess(x)

    class PostProcess(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.depth = parent.depth
            self.postprocess = nn.Sequential(
                nn.Conv2d(in_channels=8*self.depth, out_channels=8*self.depth, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(8*self.depth),
                nn.ReLU(),
                nn.Conv2d(in_channels=8 * self.depth, out_channels=8 * self.depth, kernel_size=5, stride=1, padding=2),
                nn.InstanceNorm2d(8 * self.depth),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8*self.depth, out_channels=1, kernel_size=5, stride=1, padding=2, output_padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.ReLU()
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
            self.Conv_16_16 = nn.Conv2d(in_channels=16*self.depth, out_channels=16*self.depth, kernel_size=1, stride=1)
            self.Conv_16_32 = nn.Conv2d(in_channels=16*self.depth, out_channels=32*self.depth, kernel_size=5, stride=2, padding=2)
            self.Conv_32_32 = nn.Conv2d(in_channels=32*self.depth, out_channels=32*self.depth, kernel_size=1, stride=1)
            self.Conv_32_64 = nn.Conv2d(in_channels=32*self.depth, out_channels=64*self.depth, kernel_size=5, stride=2, padding=2)
            self.Conv_64_64 = nn.Conv2d(in_channels=64*self.depth, out_channels=64*self.depth, kernel_size=1, stride=1)
            self.Conv_64_128 = nn.Conv2d(in_channels=64*self.depth, out_channels=128*self.depth, kernel_size=5, stride=2, padding=2)
            self.Conv_128_128 = nn.Conv2d(in_channels=128*self.depth, out_channels=128*self.depth, kernel_size=1, stride=1)
        def forward(self, x):
            enc_8 = x
            enc_16 = self.Conv_16_16(self.activ(self.IN_16(self.Conv_8_16(x))))
            enc_32 = self.Conv_32_32(self.activ(self.IN_32(self.Conv_16_32(enc_16))))
            enc_64 = self.Conv_64_64(self.activ(self.IN_64(self.Conv_32_64(enc_32))))
            enc_128 = self.Conv_128_128(self.activ(self.IN_128(self.Conv_64_128(enc_64))))
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
            self.Deconv_64_64 = nn.Conv2d(in_channels=64*self.depth, out_channels=64*self.depth, kernel_size=1)
            self.Deconv_64_32 = nn.ConvTranspose2d(in_channels=(64+64)*self.depth, out_channels=32*self.depth,
                                                   kernel_size=5, stride=2, padding=2, output_padding=1)
            self.Deconv_32_32 = nn.Conv2d(in_channels=32*self.depth, out_channels=32*self.depth, kernel_size=1)
            self.Deconv_32_16 = nn.ConvTranspose2d(in_channels=(32+32)*self.depth, out_channels=16*self.depth,
                                                   kernel_size=5, stride=2, padding=2, output_padding=1)
            self.Deconv_16_16 = nn.Conv2d(in_channels=16*self.depth, out_channels=16*self.depth, kernel_size=1)
            self.Deconv_16_8 = nn.ConvTranspose2d(in_channels=(16+16)*self.depth, out_channels=8*self.depth,
                                                  kernel_size=5, stride=2, padding=2, output_padding=1)
            self.Deconv_8_8 = nn.Conv2d(in_channels=8*self.depth, out_channels=8*self.depth, kernel_size=1)

        def forward(self, enc_8, enc_16, enc_32, enc_64, enc_128):
            dec_64 = self.Deconv_64_64(self.activ(self.IN_64(self.Deconv_128_64(enc_128))))
            dec_32 = self.Deconv_32_32(self.activ(self.IN_32(self.Deconv_64_32(torch.cat([enc_64, dec_64], dim=1)))))
            dec_16 = self.Deconv_16_16(self.activ(self.IN_16(self.Deconv_32_16(torch.cat([enc_32, dec_32], dim=1)))))
            dec_8 = self.Deconv_8_8(self.activ(self.IN_8(self.Deconv_16_8(torch.cat([enc_16, dec_16], dim=1)))))
            return dec_8

    def __init__(self, guess, alpha, depth, input_masking=False):
        super().__init__()
        self.guess = alpha*guess
        self.depth = depth
        self.noise = 5*guess.abs().mean()*torch.randn(*guess.shape).to(device=REC_CONFIG['DEVICE'])*16e-9
        self.activ = nn.ReLU()
        self.IN_8 = nn.InstanceNorm2d(8*self.depth)
        self.IN_16 = nn.InstanceNorm2d(16*self.depth)
        self.IN_32 = nn.InstanceNorm2d(32*self.depth)
        self.IN_64 = nn.InstanceNorm2d(64*self.depth)
        self.IN_128 = nn.InstanceNorm2d(128*self.depth)
        self.preprocess = NV_Net.PreProcess(self)
        self.encode = NV_Net.Encoder(self)
        self.decode_x = NV_Net.Decoder(self)
        self.decode_y = NV_Net.Decoder(self)
        self.decode_z = NV_Net.Decoder(self)
        self.postprocess_x = NV_Net.PostProcess(self)
        self.postprocess_y = NV_Net.PostProcess(self)
        self.postprocess_z = NV_Net.PostProcess(self)

    def forward(self, x):
        input = self.preprocess(x)
        encoded = self.encode(input)
        decoded_x = self.decode_x(*encoded)
        decoded_y = self.decode_y(*encoded)
        decoded_z = self.decode_z(*encoded)
        output_x = self.postprocess_x(decoded_x)
        output_y = self.postprocess_y(decoded_y)
        output_z = self.postprocess_z(decoded_z)
        return (torch.cat((output_x, output_y, output_z), dim=1))*MASK