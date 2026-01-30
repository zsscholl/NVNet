from NRTA.backend.packages import *

class PlotML3d():
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
        self.mag_x.set_clim(vmin=np.min(x_mag_plot), vmax=np.max(x_mag_plot))
        self.rec_x.set_clim(vmin=np.min(x_rec_plot), vmax=np.max(x_rec_plot))

        self.mag_y.set_data(y_mag_plot)
        self.rec_y.set_data(y_rec_plot)
        self.mag_y.set_clim(vmin=np.min(y_mag_plot), vmax=np.max(y_mag_plot))
        self.rec_y.set_clim(vmin=np.min(y_rec_plot), vmax=np.max(y_rec_plot))

        self.mag_z.set_data(z_mag_plot)
        self.rec_z.set_data(z_rec_plot)
        self.mag_z.set_clim(vmin=np.max(z_mag_plot), vmax=np.max(z_mag_plot))
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

def toNumpy(tensor):
    return np.squeeze(tensor.cpu().detach().numpy())

def toTorch(matrix):
    return torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)

def force_norm(tensor):
    reshaped = tensor.view(1, 3, -1)
    chan_min = reshaped.min(dim=2, keepdim=True)[0]
    chan_max = reshaped.max(dim=2, keepdim=True)[0]
    eps = 1e-15
    range = chan_max - chan_min + eps
    normalized = (reshaped-chan_min)/range
    normalized = 2*normalized - 1
    return normalized.view(*tensor.shape)

def tessellate(tensor, n):
    _, _, length, width = tensor.shape
    target = torch.zeros(1, 3, n*length, n*width)
    for i in range(n):
        for j in range(n):
            x_start = i*length
            x_end = (i+1)*length
            y_start = j*length
            y_end = (j+1)*length
            target[:, :, x_start:x_end, y_start:y_end] = tensor[:, :, :, :]
    return target

def png_to_mask(filepath):
    png = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/256
    return toTorch(png)

def finite_difference_loss(tensor):
    xdiff = torch.diff(tensor, dim=0)