from dipNV.backend.utils import *
from dipNV.backend.packages import *

class forwardModel(nn.Module):
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
        self.filter = torch.where(self.k>=self.kcut, 1/(1+(self.k/self.kcut)**4), torch.ones_like(self.k))
        self.analyticFilter = torch.where(self.k>=self.testcut, 1/(1+(self.k/self.testcut)**4), torch.ones_like(self.k))
        self.nv_x = np.cos(self.CONFIG['NV']['THETA'])*np.sin(self.CONFIG['NV']['PHI'])
        self.nv_y = np.sin(self.CONFIG['NV']['THETA'])*np.sin(self.CONFIG['NV']['PHI'])
        self.nv_z = np.cos(self.CONFIG['NV']['PHI'])

        mu = 4 * torch.pi * 1e-7
        self.exp_factor = -(mu/2)*self.CONFIG['MAT_PARAMS']['THICKNESS']*torch.exp(-self.k*self.CONFIG['NV']['STANDOFF']).to(device=loaded.device)
        # self.exp_factor = 0.5*mu*self.CONFIG['MAT_PARAMS']['THICKNESS']*torch.exp(-self.k*self.CONFIG['NV']['STANDOFF']).to(self.device)
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

    def deprojectNV(self):
        x_prop = 1j*self.kx/(1j*self.nv_x*self.kx + 1j*self.nv_y*self.ky+self.nv_z*self.k)
        y_prop = 1j*self.ky/(1j*self.nv_x*self.kx + 1j*self.nv_y*self.ky+self.nv_z*self.k)
        z_prop = -self.k/(1j*self.nv_x*self.kx + 1j*self.nv_y*self.ky+self.nv_z*self.k)
        bx = torch.nan_to_num(self.nv_fft*x_prop)*self.filter
        by = torch.nan_to_num(self.nv_fft*y_prop)*self.filter
        bz = torch.nan_to_num(self.nv_fft*z_prop)*self.filter
        bx = nn.functional.interpolate(torch.real(torch.fft.ifft2(bx)).to(dtype=torch.float32), scale_factor=0.5)
        by = nn.functional.interpolate(torch.real(torch.fft.ifft2(by)).to(dtype=torch.float32), scale_factor=0.5)
        bz = nn.functional.interpolate(torch.real(torch.fft.ifft2(bz)).to(dtype=torch.float32), scale_factor=0.5)
        b_vec = torch.cat([bx, by, bz], dim=1)
        return b_vec

    def propagateMag(self, mag_tensor):
        # padded = torch.nn.functional.pad(mag_tensor, (mag_tensor.shape[-1]//2,)*4, mode='reflect')
        # fft_data = torch.fft.fft2(padded)
        fft_data = torch.tile(torch.fft.fft2(mag_tensor), (2, 2)) #*self.CONFIG['MAT_PARAMS']['THICKNESS']
        stray = self.forward_matrix.permute(2, 3, 0, 1) @ fft_data.permute(0, 2, 3, 1).unsqueeze(-1)
        stray = stray.squeeze(-1).permute(0, 3, 1, 2)*self.filter.unsqueeze(0).unsqueeze(0).detach()
        output = torch.real(torch.fft.ifft2(stray)).to(dtype=torch. float32)
        output = nn.functional.interpolate(output, scale_factor=0.5)
        # output = output[:, :, 256:768, 256:768]
        return output

    def analytic_inplane(self, stray_tensor):
        stray_tensor = torch.fft.fft2(nn.functional.interpolate(stray_tensor, scale_factor=2))
        mx_prop = -1j*self.kx/(self.exp_factor * (1j*self.kx*self.nv_x + 1j*self.nv_y*self.ky+self.nv_z*self.k))
        my_prop = -1j*self.ky/(self.exp_factor * (1j*self.kx*self.nv_x + 1j*self.nv_y*self.ky+self.nv_z*self.k))
        mx = torch.nan_to_num(stray_tensor[:, 0:1, :, :]*mx_prop)*self.filter
        my = torch.nan_to_num(stray_tensor[:, 1:2, :, :]*my_prop)*self.filter
        mx = nn.functional.interpolate(torch.real(torch.fft.ifft2(mx)).to(dtype=torch.float32), scale_factor=0.5)
        my = nn.functional.interpolate(torch.real(torch.fft.ifft2(my)).to(dtype=torch.float32), scale_factor=0.5)
        return torch.cat([mx, my, torch.zeros_like(my)], dim=1)

    class regularizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.reg_param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input):
            return input / (1 + self.reg_param*input.abs())

    def fancy_inplane(self, stray_tensor, epochs, init_lr, refrate):
        stray = torch.fft.fft2(nn.functional.interpolate(stray_tensor, scale_factor=2))
        denom = (self.exp_factor * (1j*self.kx*self.nv_x + 1j*self.nv_y*self.ky+self.nv_z*self.k))
        model = forwardModel.regularizer()
        lossfn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=init_lr)
        for epoch in tqdm(range(epochs)):
            optim.zero_grad()
            reg_denom = model(denom)
            mx_prop = -1j*self.kx / reg_denom
            my_prop = -1j * self.ky / reg_denom
            mx = stray[:, 0:1, :, :] * mx_prop
            my = stray[:, 1:2, :, :] * my_prop
            mx = nn.functional.interpolate(torch.real(torch.fft.ifft2(mx)).to(dtype=torch.float32), scale_factor=0.5)
            my = nn.functional.interpolate(torch.real(torch.fft.ifft2(my)).to(dtype=torch.float32), scale_factor=0.5)
            m_vec = torch.cat([mx, my, torch.zeros_like(my)], dim=1)
            prop_b_vec = forwardModel.propagateMag(self, m_vec)
            loss = lossfn(stray_tensor, prop_b_vec)
            if epoch % refrate == 0:
                print(f"Loss: {loss.item()}")
            loss.backward()
            optim.step()

        return m_vec
