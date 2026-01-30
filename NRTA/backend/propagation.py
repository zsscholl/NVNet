import torch

from NRTA.backend.packages import *
from NRTA.backend.utils import *

class propagator(nn.Module):
    def __init__(self, data):
        super().__init__()
        # Creating the k-space grid
        self.dx, self.dy = data.CONFIG['DX'], data.CONFIG['DY']
        self.shape = data.CONFIG['SHAPE']
        kx = torch.fft.fftfreq(self.shape, device=data.CONFIG['DEVICE'])*2*torch.pi/self.dx
        ky = torch.fft.fftfreq(self.shape, device=data.CONFIG['DEVICE'])*2*torch.pi/self.dy
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')
        self.k = torch.sqrt(self.kx**2 + self.ky**2)
        self.k = torch.clamp(self.k, min=data.CONFIG['K_MIN'], max=data.CONFIG['K_MAX'])

        # Extracting the NV parameters
        self.nv_theta, self.nv_phi = data.CONFIG['NV_THETA'], data.CONFIG['NV_PHI']
        self.nv_x = np.cos(self.nv_phi)*np.sin(self.nv_theta)
        self.nv_y = np.sin(self.nv_phi)*np.sin(self.nv_theta)
        self.nv_z = np.cos(self.nv_theta)

        # Creating the transformation matrix
        mu = 4 * torch.pi * 1e-7
        exp_factor = -(mu/2)*torch.exp(-self.k*data.CONFIG['STANDOFF']).to(device=data.CONFIG['DEVICE'],
                                                                           dtype=torch.complex64)
        d_matrix = torch.zeros((3, 3, self.shape, self.shape), device=data.CONFIG['DEVICE'],
                               dtype=torch.complex64)
        d_matrix[0, 0, :, :] = (self.kx ** 2) / self.k
        d_matrix[0, 1, :, :] = (self.kx * self.ky) / self.k
        d_matrix[0, 2, :, :] = 1j * self.kx
        d_matrix[1, 0, :, :] = self.kx * self.ky / self.k
        d_matrix[1, 1, :, :] = (self.ky ** 2) / self.k
        d_matrix[1, 2, :, :] = 1j * self.ky
        d_matrix[2, 0, :, :] = 1j * self.kx
        d_matrix[2, 1, :, :] = 1j * self.ky
        d_matrix[2, 2, :, :] = -self.k
        self.matrix = exp_factor * d_matrix

    def deproject_nv(self, nv_scan):
        nv_fft = torch.fft.fft2(nv_scan)
        x_prop = 1 / (self.nv_x + self.nv_y*self.ky/self.kx +1j*self.nv_z*self.k/self.kx)
        y_prop = 1 / (self.nv_x + self.nv_y * self.kx / self.ky + 1j * self.nv_z * self.k / self.ky)
        z_prop = 1 / (-1j*self.nv_x*self.kx/self.k -1j * self.nv_y * self.ky / self.k + self.nv_z)
        bx = torch.nan_to_num(nv_fft*x_prop)
        by = torch.nan_to_num(nv_fft*y_prop)
        bz = torch.nan_to_num(nv_fft*z_prop)
        b_fft = torch.cat([bx, by, bz], dim=1)
        return torch.real(torch.fft.ifft2(b_fft)).to(dtype=torch.float32)

    def propagate_mag(self, mag):
        fft_data = torch.fft.fft2(mag)
        stray = self.matrix.permute(2, 3, 0, 1) @ fft_data.permute(0, 2, 3, 1).unsqueeze(-1)
        stray = stray.squeeze(-1).permute(0, 3, 1, 2)
        output = torch.fft.ifft2(stray).to(dtype=torch.float32)
        return output

