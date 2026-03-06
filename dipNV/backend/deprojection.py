from dipNV.backend.utils import *
from dipNV.backend.packages import *

class deprojector(nn.Module):
    def __init__(self, loaded):
        super().__init__()
        self.raw = loaded.nv_data
        self.nv_fft = torch.tile(torch.fft.fft2(self.raw), (2, 2))
        self.device = loaded.device
        self.CONFIG = loaded.CONFIG
        kx = 2 * torch.pi * torch.fft.fftfreq(self.nv_fft.shape[-1], device=loaded.device) / self.CONFIG['DX']
        ky = 2 * torch.pi * torch.fft.fftfreq(self.nv_fft.shape[-1], device=loaded.device) / self.CONFIG['DX']
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')
        self.k = torch.sqrt(self.kx ** 2 + self.ky ** 2)
        self.k = torch.clamp(self.k, min=self.CONFIG['K_MIN'])
        self.nv_x = np.cos(self.CONFIG['NV']['THETA']) * np.sin(self.CONFIG['NV']['PHI'])
        self.nv_y = np.sin(self.CONFIG['NV']['THETA']) * np.sin(self.CONFIG['NV']['PHI'])
        self.nv_z = np.cos(self.CONFIG['NV']['PHI'])

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
        model = deprojector.regularizer()
        lossfn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=init_lr)
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

        return b_vec.detach()

