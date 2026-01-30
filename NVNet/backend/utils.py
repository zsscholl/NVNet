from NVNet.backend.packages import *

class diverg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m):
        mx = m[:, 0:1, :, :]
        my = m[:, 1:2, :, :]
        dmx_dx = torch.diff(mx, dim=3)
        dmy_dy = torch.diff(my, dim=3)
        min_h = min(dmx_dx.shape[2], dmy_dy.shape[2])
        min_w = min(dmx_dx.shape[3], dmy_dy.shape[3])
        divergence = dmx_dx[:, :, :min_h, :min_w] + dmy_dy[:, :, :min_h, :min_w]
        return torch.mean(divergence ** 2)

class PlotML3d():
    def __init__(self, input):
        self.fig, self.ax = plt.subplots(3, 3)
        self.stray_x = self.ax[0, 0].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr')
        self.mag_x = self.ax[0, 1].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr')
        self.rec_x = self.ax[0, 2].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr')
        self.stray_y = self.ax[1, 0].imshow(input[0, 1].cpu().detach().numpy(), cmap='bwr')
        self.mag_y = self.ax[1, 1].imshow(input[0, 1].cpu().detach().numpy(), cmap='bwr')
        self.rec_y = self.ax[1, 2].imshow(input[0, 1].cpu().detach().numpy(), cmap='bwr')
        self.stray_z = self.ax[2, 0].imshow(input[0, 2].cpu().detach().numpy(), cmap='bwr')
        self.mag_z = self.ax[2, 1].imshow(input[0, 2].cpu().detach().numpy(), cmap='bwr')
        self.rec_z = self.ax[2, 2].imshow(input[0, 2].cpu().detach().numpy(), cmap='bwr')
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
        self.fig.tight_layout(pad=2)

    def Render(self, model, propagated):
        self.mag_x.set_data(model[0,0].cpu().detach().numpy())
        self.rec_x.set_data(propagated[0,0].cpu().detach().numpy())
        self.mag_x.set_clim(vmin=model[0,0].min(), vmax=model[0,0].max())
        self.rec_x.set_clim(vmin=propagated[0,0].min(), vmax=propagated[0,0].max())
        self.mag_y.set_data(model[0, 1].cpu().detach().numpy())
        self.rec_y.set_data(propagated[0, 1].cpu().detach().numpy())
        self.mag_y.set_clim(vmin=model[0,1].min(), vmax=model[0,1].max())
        self.rec_y.set_clim(vmin=propagated[0,1].min(), vmax=propagated[0,1].max())
        self.mag_z.set_data(model[0, 2].cpu().detach().numpy())
        self.rec_z.set_data(propagated[0, 2].cpu().detach().numpy())
        self.mag_z.set_clim(vmin=model[0,2].min(), vmax=model[0,2].max())
        self.rec_z.set_clim(vmin=propagated[0,2].min(), vmax=propagated[0,2].max())
        self.cbar_mag_x.update_normal(self.mag_x)
        self.cbar_rec_x.update_normal(self.rec_x)
        self.cbar_mag_y.update_normal(self.mag_y)
        self.cbar_rec_y.update_normal(self.rec_y)
        self.cbar_mag_z.update_normal(self.mag_z)
        self.cbar_rec_z.update_normal(self.rec_z)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def Save(self, epoch):
        # self.mag_x.set_data(model[0, 0].cpu().detach().numpy())
        # self.rec_x.set_data(propagated[0, 0].cpu().detach().numpy())
        # self.mag_x.set_clim(vmin=model[0, 0].min(), vmax=model[0, 0].max())
        # self.rec_x.set_clim(vmin=propagated[0, 0].min(), vmax=propagated[0, 0].max())
        # self.mag_y.set_data(model[0, 1].cpu().detach().numpy())
        # self.rec_y.set_data(propagated[0, 1].cpu().detach().numpy())
        # self.mag_y.set_clim(vmin=model[0, 1].min(), vmax=model[0, 1].max())
        # self.rec_y.set_clim(vmin=propagated[0, 1].min(), vmax=propagated[0, 1].max())
        # self.mag_z.set_data(model[0, 2].cpu().detach().numpy())
        # self.rec_z.set_data(propagated[0, 2].cpu().detach().numpy())
        # self.mag_z.set_clim(vmin=model[0, 2].min(), vmax=model[0, 2].max())
        # self.rec_z.set_clim(vmin=propagated[0, 2].min(), vmax=propagated[0, 2].max())
        # self.cbar_mag_x.update_normal(self.mag_x)
        # self.cbar_rec_x.update_normal(self.rec_x)
        # self.cbar_mag_y.update_normal(self.mag_y)
        # self.cbar_rec_y.update_normal(self.rec_y)
        # self.cbar_mag_z.update_normal(self.mag_z)
        # self.cbar_rec_z.update_normal(self.rec_z)
        # self.cbar_stray_x.ax.tick_params(labelsize=8)
        # self.cbar_stray_y.ax.tick_params(labelsize=8)
        # self.cbar_stray_z.ax.tick_params(labelsize=8)
        # self.cbar_mag_x.ax.tick_params(labelsize=8)
        # self.cbar_mag_y.ax.tick_params(labelsize=8)
        # self.cbar_mag_z.ax.tick_params(labelsize=8)
        # self.cbar_rec_x.ax.tick_params(labelsize=8)
        # self.cbar_rec_y.ax.tick_params(labelsize=8)
        # self.cbar_rec_z.ax.tick_params(labelsize=8)
        # self.fig.tight_layout(pad=2)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        self.fig.savefig(f'epoch_{epoch}.png', dpi=1200, bbox_inches='tight',
                         facecolor='white', edgecolor='none')

def toNumpy(tensor):
    return np.squeeze(tensor.cpu().detach().numpy())

def toTorch(matrix):
    return torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)

def plot_magnetic_field_map(B_map, title, cmap='bwr'):
    plt.figure(figsize=(8, 6))
    plt.imshow(B_map, cmap=cmap, aspect='auto')
    plt.colorbar(label="units")
    plt.title(title)
    plt.xlabel("Scan Pixel (X)")
    plt.ylabel("Scan Line (Y)")
    plt.show()