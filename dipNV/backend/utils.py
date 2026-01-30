from dipNV.backend.packages import *

def toTorch(matrix):
    return torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)

def toNumpy(tensor):
    return np.squeeze(tensor.cpu().detach().numpy())

class PlotML():
    def __init__(self, input):
        input = input.to(dtype=torch.float32)
        _, _, len_x, len_y = input.shape
        self.fig, self.ax = plt.subplots(3, 3)

        self.stray_x = self.ax[0, 0].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr', origin='lower')
        self.mag_x = self.ax[0, 1].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr', origin='lower')
        self.rec_x = self.ax[0, 2].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr', origin='lower')

        self.stray_y = self.ax[1, 0].imshow(input[0, 1].cpu().detach().numpy(), cmap='bwr', origin='lower')
        self.mag_y = self.ax[1, 1].imshow(input[0, 1].cpu().detach().numpy(), cmap='bwr', origin='lower')
        self.rec_y = self.ax[1, 2].imshow(input[0, 1].cpu().detach().numpy(), cmap='bwr', origin='lower')

        self.stray_z = self.ax[2, 0].imshow(input[0, 2].cpu().detach().numpy(), cmap='bwr', origin='lower')
        self.mag_z = self.ax[2, 1].imshow(input[0, 2].cpu().detach().numpy(), cmap='bwr', origin='lower')
        self.rec_z = self.ax[2, 2].imshow(input[0, 2].cpu().detach().numpy(), cmap='bwr', origin='lower')

        self.cbar_stray_x = self.fig.colorbar(self.stray_x, ax=self.ax[0, 0], shrink=0.8)
        self.cbar_mag_x = self.fig.colorbar(self.mag_x, ax=self.ax[0, 1], shrink=0.8)
        self.cbar_rec_x = self.fig.colorbar(self.rec_x, ax=self.ax[0, 2], shrink=0.8)

        self.cbar_stray_y = self.fig.colorbar(self.stray_y, ax=self.ax[1, 0], shrink=0.8)
        self.cbar_mag_y = self.fig.colorbar(self.mag_y, ax=self.ax[1, 1], shrink=0.8)
        self.cbar_rec_y = self.fig.colorbar(self.rec_y, ax=self.ax[1, 2], shrink=0.8)

        self.cbar_stray_z = self.fig.colorbar(self.stray_z, ax=self.ax[2, 0], shrink=0.8)
        self.cbar_mag_z = self.fig.colorbar(self.mag_z, ax=self.ax[2, 1], shrink=0.8)
        self.cbar_rec_z = self.fig.colorbar(self.rec_z, ax=self.ax[2, 2], shrink=0.8)

        # self.ax[0, 0].set_title(r'Simulated $\vec{B}$')
        # self.ax[0, 1].set_title(r'Reconstructed $\vec{M}$')
        # self.ax[0, 2].set_title(r'Propagated $\vec{B}$')
        self.x_grid, self.y_grid = torch.meshgrid(torch.arange(len_x), torch.arange(len_y), indexing='xy')
        self.quiver_z = None

        self.fig.tight_layout(pad=2)

    def Render(self, model, propagated, do_quiver=False):
        x_mag_plot = model[0,0].cpu().detach().numpy()
        x_rec_plot = propagated[0,0].cpu().detach().numpy()
        y_mag_plot = model[0, 1].cpu().detach().numpy()
        y_rec_plot = propagated[0, 1].cpu().detach().numpy()
        z_mag_plot = model[0, 2].cpu().detach().numpy()
        z_rec_plot = propagated[0, 2].cpu().detach().numpy()

        self.mag_x.set_data(x_mag_plot)
        self.rec_x.set_data(x_rec_plot)
        self.mag_x.set_clim(vmin=-np.max(np.abs(x_mag_plot)), vmax=np.max(np.abs(x_mag_plot)))
        self.rec_x.set_clim(vmin=np.min(x_rec_plot), vmax=np.max(x_rec_plot))

        self.mag_y.set_data(y_mag_plot)
        self.rec_y.set_data(y_rec_plot)
        self.mag_y.set_clim(vmin=-np.max(np.abs(y_mag_plot)), vmax=np.max(np.abs(y_mag_plot)))
        self.rec_y.set_clim(vmin=np.min(y_rec_plot), vmax=np.max(y_rec_plot))

        self.mag_z.set_data(z_mag_plot)
        self.rec_z.set_data(z_rec_plot)
        self.mag_z.set_clim(vmin=-np.max(np.abs(z_mag_plot)), vmax=np.max(np.abs(z_mag_plot)))
        self.rec_z.set_clim(vmin=np.min(z_rec_plot), vmax=np.max(z_rec_plot))

        x_mag_texture = x_mag_plot/np.where(np.abs(x_mag_plot) <= 1e-8, 1e-8, np.abs(x_mag_plot))
        y_mag_texture = y_mag_plot/np.where(np.abs(y_mag_plot) <= 1e-8, 1e-8, np.abs(y_mag_plot))
        if do_quiver is True:
            if self.quiver_z is not None:
                self.quiver_z.remove()
            step = 80
            self.quiver_z = self.ax[2, 1].quiver(
                self.x_grid[::step, ::step], self.y_grid[::step, ::step],
                x_mag_texture[::step, ::step], y_mag_texture[::step, ::step], color='red')

        self.cbar_mag_x.update_normal(self.mag_x)
        self.cbar_rec_x.update_normal(self.rec_x)

        self.cbar_mag_y.update_normal(self.mag_y)
        self.cbar_rec_y.update_normal(self.rec_y)

        self.cbar_mag_z.update_normal(self.mag_z)
        self.cbar_rec_z.update_normal(self.rec_z)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

class divergence(nn.Module):
    def __init__(self, loaded):
        super().__init__()
        self.dx = loaded.CONFIG['DX']

    def forward(self,  tensor):
        x_contribution = torch.diff(tensor[0, 0, :, :])/(self.dx)
        y_contribution = torch.diff(tensor[0, 1, :, :])/(self.dx)
        z_contribution = torch.diff(tensor[0, 2, :, :])/(self.dx)
        div = x_contribution + y_contribution + z_contribution
        return torch.mean(div**2)

def create_diagonal_mask(size, angle):
    array = np.zeros((size, size))
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(angle)

    X_rot = X*np.cos(theta) - Y*np.sin(theta)
    Y_rot = X*np.sin(theta) + Y*np.cos(theta)

    array[(Y_rot > X_rot) & (Y_rot > -X_rot)] = 1
    array[(Y_rot > X_rot) & (Y_rot < -X_rot)] = -1
    array[(Y_rot < X_rot) & (Y_rot < -X_rot)] = 1
    array[(Y_rot < X_rot) & (Y_rot > -X_rot)] = -1

    return array

def transform_mask(tensor_mask, scale, rotation):
    tensor_mask_x = nn.functional.interpolate(tensor_mask[:, 0:1, :, :], scale_factor=scale)
    tensor_mask_y = nn.functional.interpolate(tensor_mask[:, 1:2, :, :], scale_factor=scale)
    tensor_mask_z = nn.functional.interpolate(tensor_mask[:, 2:, :, :], scale_factor=scale)

    width = int(tensor_mask.shape[-1])
    delta = int(tensor_mask_x.shape[-1] - width)
    if delta <0:
        delta = -delta
    output = torch.cat([tensor_mask_x, tensor_mask_y, tensor_mask_z], dim=1)
    output = tv.transforms.functional.rotate(output, rotation)
    if scale != 1:
        output = output[:, :, delta//2:delta//2+width, delta//2:delta//2+width]
    return output




