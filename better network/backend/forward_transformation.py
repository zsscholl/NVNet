import numpy as np
import torch
import torch.nn as nn
import scipy as scp
from .config import *

class ForwardTransform(nn.Module):
    def __init__(self, pixel_scale):
        super().__init__()
        self.kx, self.ky, self.k, self.matrix = None, None, None, None
        self.pixel_scale = pixel_scale

    def InitializeKSpace(self):
        if self.k is not None: return
        scaling_factor = 2 * torch.pi / self.pixel_scale

        kx_vec = scaling_factor * torch.fft.fftfreq(self.pixel_scale, device=REC_CONFIG['DEVICE'])
        ky_vec = scaling_factor * torch.fft.fftfreq(self.pixel_scale, device=REC_CONFIG['DEVICE'])

        self.kx, self.ky = torch.meshgrid(kx_vec, ky_vec, indexing='xy')

        self.k = torch.sqrt(self.kx ** 2 + self.ky ** 2)
        self.k = torch.clamp(self.k, min=REC_CONFIG['PROP_PARAMS']['K_EPS'])

    def NVtoStray(self, nv):
        self.InitializeKSpace()
        kx, ky, k = self.k, self.kx, self.ky

        ex = np.cos(DATA_CONFIG['NV_PARAMS']['THETA'])*np.sin(DATA_CONFIG['NV_PARAMS']['PHI'])
        ey = np.sin(DATA_CONFIG['NV_PARAMS']['THETA'])*np.sin(DATA_CONFIG['NV_PARAMS']['PHI'])
        ez = np.cos(DATA_CONFIG['NV_PARAMS']['PHI'])
        denom = (1j*kx*ex+1j*ky*ey-k*ez)

        nv_fft = torch.fft.fft2(nv)
        bx_fft = 1j*kx*nv_fft/denom
        by_fft = 1j*ky*nv_fft/denom
        bz_fft = -k*nv_fft/denom

        bx_real = torch.fft.ifft2(bx_fft).real
        by_real = torch.fft.ifft2(by_fft).real
        bz_real = torch.fft.ifft2(bz_fft).real

        return torch.cat((bx_real, by_real, bz_real), dim=1).to(dtype=torch.float32)

    def ForwardMatrix(self):
        if self.matrix is not None: return
        self.InitializeKSpace()
        kx, ky, k = self.kx, self.ky, self.k

        mu0 = 4 * np.pi * 1e-7
        exp_factor = (mu0 / 2) * torch.exp(-k * DATA_CONFIG['NV_PARAMS']['SCAN_HEIGHT'])
        exp_factor = exp_factor.to(device=REC_CONFIG['DEVICE'], dtype=torch.complex64)

        exp_factor = exp_factor.unsqueeze(0).unsqueeze(0)

        d_matrix = torch.zeros((3, 3, self.pixel_scale, self.pixel_scale), device=REC_CONFIG['DEVICE'],
                               dtype=torch.complex64)

        d_matrix[0, 0, :, :] = -(kx ** 2) / k
        d_matrix[0, 1, :, :] = -(kx * ky) / k
        d_matrix[0, 2, :, :] = -1j * kx
        d_matrix[1, 0, :, :] = -kx * ky / k
        d_matrix[1, 1, :, :] = -(ky ** 2) / k
        d_matrix[1, 2, :, :] = -1j * ky
        d_matrix[2, 0, :, :] = -1j * kx
        d_matrix[2, 1, :, :] = -1j * ky
        d_matrix[2, 2, :, :] = k

        self.matrix = exp_factor * d_matrix
        self.matrix[:, :, self.k < REC_CONFIG['PROP_PARAMS']['K_EPS']] = 0

    def StrayFromMag(self, mag_vec):
        self.ForwardMatrix()
        mag_vec_fft = torch.fft.fft2(mag_vec.to(torch.complex64), norm='ortho')
        mat = self.matrix.permute(2, 3, 0, 1)  # (H, W, 3, 3)
        mag_fft_permuted = mag_vec_fft.permute(0, 2, 3, 1).unsqueeze(-1)  # (N, H, W, 3, 1)
        stray_vec_fft_permuted = mat @ mag_fft_permuted
        stray_vec_fft = stray_vec_fft_permuted.squeeze(-1).permute(0, 3, 1, 2)
        stray_vec_real = torch.real(torch.fft.ifft2(stray_vec_fft, norm='ortho')).to(dtype=torch.float32)
        return stray_vec_real