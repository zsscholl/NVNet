import torch

from NVNet.backend.packages import *
from NVNet.backend.config import *

class ForwardTransform(nn.Module):
    def __init__(self, pixel_scale):
        super().__init__()
        self.kx, self.ky, self.k, self.matrix = None, None, None, None
        self.pixel_scale = pixel_scale
        self.nv_x, self.nv_y, self.nv_z = None, None, None

    def InitializeKSpace(self):
        if self.k is not None: return
        x_scaling_factor = 2 * torch.pi / DATA_CONFIG['X_PIX_WIDTH']
        y_scaling_factor = 2 * torch.pi / DATA_CONFIG['Y_PIX_WIDTH']

        kx_vec = x_scaling_factor * torch.fft.fftfreq(self.pixel_scale, device=REC_CONFIG['DEVICE'])
        ky_vec = y_scaling_factor * torch.fft.fftfreq(self.pixel_scale, device=REC_CONFIG['DEVICE'])

        self.kx, self.ky = torch.meshgrid(kx_vec, ky_vec, indexing='xy')

        self.k = torch.sqrt(self.kx ** 2 + self.ky ** 2)
        self.k = torch.clamp(self.k, min=REC_CONFIG['PROP_PARAMS']['K_EPS'])

    def NVtoStray(self, nv):
        self.InitializeKSpace()
        kx, ky, k = self.k, self.kx, self.ky
        self.nv_x = np.cos(DATA_CONFIG['NV_PARAMS']['PHI'])*np.sin(DATA_CONFIG['NV_PARAMS']['THETA'])
        self.nv_y = np.sin(DATA_CONFIG['NV_PARAMS']['PHI'])*np.sin(DATA_CONFIG['NV_PARAMS']['THETA'])
        self.nv_z = np.cos(DATA_CONFIG['NV_PARAMS']['THETA'])
        denom = (1j*kx*self.nv_x+1j*ky*self.nv_y-k*self.nv_z)

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
        exp_factor = -(mu0 / 2) * torch.exp(-k * DATA_CONFIG['NV_PARAMS']['SCAN_HEIGHT'])
        exp_factor = exp_factor.to(device=REC_CONFIG['DEVICE'], dtype=torch.complex64)

        exp_factor = exp_factor.unsqueeze(0).unsqueeze(0)

        d_matrix = torch.zeros((3, 3, self.pixel_scale, self.pixel_scale), device=REC_CONFIG['DEVICE'],
                               dtype=torch.complex64)

        d_matrix[0, 0, :, :] = (kx ** 2) / k
        d_matrix[0, 1, :, :] = (kx * ky) / k
        d_matrix[0, 2, :, :] = 1j * kx
        d_matrix[1, 0, :, :] = kx * ky / k
        d_matrix[1, 1, :, :] = (ky ** 2) / k
        d_matrix[1, 2, :, :] = 1j * ky
        d_matrix[2, 0, :, :] = 1j * kx
        d_matrix[2, 1, :, :] = 1j * ky
        d_matrix[2, 2, :, :] = -k

        # d_matrix = torch.nan_to_num(d_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        self.matrix = exp_factor * d_matrix
        self.matrix[:, :, self.k < REC_CONFIG['PROP_PARAMS']['K_EPS']] = 0

    def StrayFromMag(self, mag_vec):
        self.ForwardMatrix()
        window = torch.hann_window(len(self.k)).to(device=REC_CONFIG['DEVICE'])
        mag_vec = mag_vec*DATA_CONFIG['THICKNESS']
        mag_vec_fft = torch.fft.fft2(mag_vec.to(torch.complex64)) #*window
        mat = self.matrix.permute(2, 3, 0, 1)  # (H, W, 3, 3)
        mag_fft_permuted = mag_vec_fft.permute(0, 2, 3, 1).unsqueeze(-1)  # (N, H, W, 3, 1)
        stray_vec_fft_permuted = mat @ mag_fft_permuted
        stray_vec_fft = stray_vec_fft_permuted.squeeze(-1).permute(0, 3, 1, 2)
        stray_vec_real = torch.real(torch.fft.ifft2(stray_vec_fft)).to(dtype=torch.float32)
        return stray_vec_real