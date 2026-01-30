import torch

from NVNet.backend.packages import *
from NVNet.train import *

# quick_odmr = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\gwys.npy')
# IMRE = torch.from_numpy(quick_odmr[120:160, 40:80]).unsqueeze(0).unsqueeze(0).to(device=REC_CONFIG['DEVICE'])
# skyrm = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\neel_stray1.npy')
# SKYRMION = torch.from_numpy(skyrm).to(device=REC_CONFIG['DEVICE']).transpose(0, 1)[:, 0:1, 100:155, 100:155]
# test = Train(SKYRMION, 5000, is_nv=False)
# mag = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\neel_mag1.npy')
# plt.imshow(mag[0, 0, 100:155, 100:155], cmap='bwr')
# plt.show()

clover_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_stray_field.npy')[0:1, 28, :, :]
CLOVER_DATA = torch.from_numpy(clover_raw).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
landau_surface = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_state.npy')[0:3, 6, :, :]
LANDAU = torch.from_numpy(landau_surface).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
transform = ForwardTransform(256)
stray = transform.StrayFromMag(LANDAU)
plot_data = toNumpy(stray)

# fig, ax = plt.subplots(1, 2)
# rec = ax[0].imshow(plot_data[0], cmap='bwr')
# stray = ax[1].imshow(clover_raw[0], cmap='bwr')
# ax[0].set_title('Stray Field from Magnetization')
# ax[1].set_title('Saved Stray Field')
# fig.colorbar(rec, ax=ax[0], shrink=0.8)
# fig.colorbar(stray, ax=ax[1], shrink=0.8)
# plt.show()

fig, ax = plt.subplots(3, 1)
xcomp = ax[0].imshow(landau_surface[0], cmap='bwr')
ycomp = ax[1].imshow(landau_surface[1], cmap='bwr')
zcomp = ax[2].imshow(landau_surface[2], cmap='bwr')
fig.colorbar(xcomp, ax=ax[0], shrink=0.8)
fig.colorbar(ycomp, ax=ax[1], shrink=0.8)
fig.colorbar(zcomp, ax=ax[2], shrink=0.8)
plt.show()

# test = Train(CLOVER_DATA, 20000, is_nv=False)