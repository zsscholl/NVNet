import torch
import torch.nn as nn

class NVBranch(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.activ = nn.LeakyReLU()
        self.ConvMask = nn.Conv2d(in_channels=1, out_channels=8*depth, kernel_size=5, stride=2, padding=2)
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=8*depth, kernel_size=5, stride=2, padding=2)
        self.BN1 = nn.BatchNorm2d(8*depth)
        self.Conv2 = nn.Conv2d(in_channels=8*depth, out_channels=16*depth, kernel_size=5, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(16*depth)
        self.Conv3 = nn.Conv2d(in_channels=16*depth, out_channels=32*depth, kernel_size=5, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(32*depth)
        self.Conv4 = nn.Conv2d(in_channels=32*depth, out_channels=64*depth, kernel_size=5, stride=2, padding=2)
        self.BN4 = nn.BatchNorm2d(64*depth)
        self.Deconv1 = nn.ConvTranspose2d(in_channels=64*depth, out_channels=32*depth, kernel_size=5,
                                          stride=2, padding=2, output_padding=1)
        self.BN5 = nn.BatchNorm2d(32*depth)
        self.Deconv2 = nn.ConvTranspose2d(in_channels=32*depth, out_channels=16*depth, kernel_size=5,
                                          stride=2, padding=2, output_padding=1)
        self.BN6 = nn.BatchNorm2d(16*depth)
        self.Deconv3 = nn.ConvTranspose2d(in_channels=16*depth, out_channels=8*depth, kernel_size=5,
                                          stride=2, padding=2, output_padding=1)
        self.BN7 = nn.BatchNorm2d(8*depth)
        self.Deconv4 = nn.ConvTranspose2d(in_channels=8*depth, out_channels=1, kernel_size=5,
                                          stride=2, padding=2, output_padding=1)

    def forward(self, x, mask):
        roi = self.Conv1(x)
        roi = self.activ(roi)
        roi = self.BN1(roi)
        roi = self.Conv2(roi)
        roi = self.BN2(roi)
        roi = self.activ(roi)
        roi = self.Conv3(roi)
        roi = self.BN3(roi)
        roi = self.activ(roi)
        roi = self.Conv4(roi)
        roi = self.BN4(roi)
        roi = self.activ(roi)
        roi = self.Deconv1(roi)
        roi = self.BN5(roi)
        roi = self.activ(roi)
        roi = self.Deconv2(roi)
        roi = self.BN6(roi)
        roi = self.activ(roi)
        roi = self.Deconv3(roi)
        roi = self.BN7(roi)
        roi = self.activ(roi)
        roi = self.Deconv4(roi)*mask
        return roi
