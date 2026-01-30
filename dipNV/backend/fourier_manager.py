from dipNV.backend.utils import *
from dipNV.backend.packages import *

class overseer(nn.Module):
    def __init__(self, loaded):
        super().__init__()
        self.raw = loaded.nv_data
        self.nv_fft = torch.tile(torch.fft.fft2(self.raw), (2, 2))
        self.device = loaded.device
        self.CONFIG = loaded.CONFIG
        self.kcut = loaded.CONFIG['K_CUTOFF']
        self.testcut = loaded.CONFIG['TEST_K_CUTOFF']
        kx = 2*torch.pi*torch.fft.fftfreq(self.nv_fft.shape[-1], device=loaded.device)/self.CONFIG['DX']
        ky = 2*torch.pi*torch.fft.fftfreq(self.nv_fft.shape[-1], device=loaded.device)/self.CONFIG['DX']
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')
        self.k = torch.sqrt(self.kx ** 2 + self.ky ** 2)
        self.k = torch.clamp(self.k, min=self.CONFIG['K_MIN'])
        self.nv_x = np.cos(self.CONFIG['NV']['THETA'])*np.sin(self.CONFIG['NV']['PHI'])
        self.nv_y = np.sin(self.CONFIG['NV']['THETA'])*np.sin(self.CONFIG['NV']['PHI'])
        self.nv_z = np.cos(self.CONFIG['NV']['PHI'])

        mu = 4 * torch.pi * 1e-7
        self.exp_factor = -(mu/2)*self.CONFIG['MAT_PARAMS']['THICKNESS']*torch.exp(-self.k*self.CONFIG['NV']['STANDOFF']).to(device=loaded.device)
        # self.exp_factor = 0.5*mu*self.CONFIG['MAT_PARAMS']['THICKNESS']*torch.exp(-self.k*self.CONFIG['NV']['STANDOFF']).to(self.device)

        self.kx = self.kx.detach()
        self.ky = self.ky.detach()
        self.k = self.k.detach()
        self.exp_factor = self.exp_factor.detach()

        forward_matrix = torch.zeros((3, 3, self.nv_fft.shape[-1], self.nv_fft.shape[-1]), device=loaded.device,
                               dtype=torch.complex64)
        forward_matrix[0, 0, :, :] = (self.kx ** 2) / self.k
        forward_matrix[0, 1, :, :] = (self.kx * self.ky) / self.k
        forward_matrix[0, 2, :, :] = 1j * self.kx
        forward_matrix[1, 0, :, :] = self.kx * self.ky / self.k
        forward_matrix[1, 1, :, :] = (self.ky ** 2) / self.k
        forward_matrix[1, 2, :, :] = 1j * self.ky
        forward_matrix[2, 0, :, :] = 1j * self.kx
        forward_matrix[2, 1, :, :] = 1j * self.ky
        forward_matrix[2, 2, :, :] = -self.k
        self.forward_matrix = self.exp_factor * forward_matrix


    class regularizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.reg_param = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, input):
            return input / (1 + self.reg_param*input.abs())

    def reproject(self, stray_field):
        return self.nv_x * stray_field[:, 0:1, :, :] + self.nv_y * stray_field[:, 1:2, :, :] + self.nv_z * stray_field[:, 2:, :, :]

    def iterative_deprojection(self, epochs, init_lr, refrate):
        denom = 1j * self.nv_x * self.kx + 1j * self.nv_y * self.ky + self.nv_z * self.k
        model = overseer.regularizer()
        lossfn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=init_lr)
        print('Regularizing the deprojected stray field...')
        for epoch in tqdm(range(epochs)):
            optim.zero_grad()
            reg_denom = model(denom)
            x_prop = 1j * self.kx / reg_denom
            y_prop = 1j * self.ky / reg_denom
            z_prop = self.k / reg_denom
            bx = self.nv_fft * x_prop
            by = self.nv_fft * y_prop
            bz = self.nv_fft * z_prop
            bx = nn.functional.interpolate(torch.real(torch.fft.ifft2(bx)).to(dtype=torch.float32), scale_factor=0.5)
            by = nn.functional.interpolate(torch.real(torch.fft.ifft2(by)).to(dtype=torch.float32), scale_factor=0.5)
            bz = nn.functional.interpolate(torch.real(torch.fft.ifft2(bz)).to(dtype=torch.float32), scale_factor=0.5)
            b_vec = torch.cat([bx, by, bz], dim=1)

            fake_nv = self.reproject(b_vec).squeeze()
            loss = lossfn(self.raw.squeeze(), fake_nv)
            if epoch % refrate == 0:
                print(f"Loss: {loss.item()}")
            loss.backward()
            optim.step()

        return b_vec
    def propagateMag(self, mag_tensor):
        # padded = torch.nn.functional.pad(mag_tensor, (mag_tensor.shape[-1]//2,)*4, mode='reflect')
        # fft_data = torch.fft.fft2(padded)
        fft_data = torch.tile(torch.fft.fft2(mag_tensor), (2, 2)) #*self.CONFIG['MAT_PARAMS']['THICKNESS']
        stray = self.forward_matrix.permute(2, 3, 0, 1) @ fft_data.permute(0, 2, 3, 1).unsqueeze(-1)
        stray = stray.squeeze(-1).permute(0, 3, 1, 2)
        output = torch.real(torch.fft.ifft2(stray)).to(dtype=torch.float32)
        output = nn.functional.interpolate(output, scale_factor=0.5)
        # output = output[:, :, 256:768, 256:768]
        return output

    def iterative_analytic(self, stray_tensor, epochs, init_lr, refrate):
        k_mask = (self.k > self.CONFIG['K_MIN']) & (self.k < self.kcut)
        k_mask = k_mask.bool()
        stray_tensor = stray_tensor.detach()
        stray = torch.fft.fft2(nn.functional.interpolate(stray_tensor, scale_factor=2))
        denom = (
                self.exp_factor *
                (1j * self.kx * self.nv_x + 1j * self.nv_y * self.ky + self.nv_z * self.k)
        ).detach()
        denom = denom * k_mask
        model = overseer.regularizer()
        lossfn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=init_lr)
        print('Regularizing the analytically recovered magnetization')
        for epoch in tqdm(range(epochs)):
            optim.zero_grad()
            reg_denom = model(denom)
            mx_prop = torch.zeros_like(reg_denom)
            my_prop = torch.zeros_like(reg_denom)
            mx_prop[k_mask] = -1j * self.kx[k_mask] / reg_denom[k_mask]
            my_prop[k_mask] = 1j * self.ky[k_mask] / reg_denom[k_mask]
            mx = mx_prop*stray[:, 0:1, :, :]
            my = my_prop*stray[:, 1:2, :, :]
            mx = nn.functional.interpolate(torch.real(torch.fft.ifft2(mx)).to(dtype=torch.float32), scale_factor=0.5)
            my = nn.functional.interpolate(torch.real(torch.fft.ifft2(my)).to(dtype=torch.float32), scale_factor=0.5)
            m_vec = torch.cat([mx, my, torch.zeros_like(my)], dim=1)
            # m_vec = torch.clamp(m_vec, min=-self.CONFIG['MAT_PARAMS']['M_SAT'], max=+self.CONFIG['MAT_PARAMS']['M_SAT'])
            prop_stray = self.propagateMag(m_vec)
            loss = lossfn(stray_tensor, prop_stray)
            if epoch % refrate == 0:
                print(f"Loss: {loss.item()}")
            loss.backward()
            optim.step()
            # test
        return m_vec