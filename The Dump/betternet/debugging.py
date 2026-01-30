from betternet.dual_train import *
from train import *

raw = np.load(r'/The Dump/betternet/data/cloverstray.npy')
TORCH_DATA = torch.from_numpy(raw).transpose(0, 1).to(device=REC_CONFIG['DEVICE'])
clover = np.load(r'/The Dump/betternet/data/clovermag.npy')
TORCH_MAG = torch.from_numpy(clover).transpose(0, 1).to(device=REC_CONFIG['DEVICE'])
TORCH_MAG[:, 1:2, :, :] = torch.zeros_like(TORCH_MAG[:, 1:2, :, :])

CLOVERMASK = torch.where(TORCH_MAG[:, 0, :, :].abs() +TORCH_MAG[:, 0, :, :].abs() <= 1e-5, torch.zeros_like(TORCH_MAG[:, 0, :, :]), torch.ones_like(TORCH_MAG[:, 0, :, :]) )
CLOVERMASK = CLOVERMASK.unsqueeze(0)
np.save('BINARY_MASK.npy', toNumpy(CLOVERMASK))

# transform = ForwardTransform(256)
# rebuild = transform.StrayFromMag(TORCH_MAG)
# rec_plot = toNumpy(rebuild)
# stray_plot = toNumpy(TORCH_DATA)
# plot_magnetic_field_map(rec_plot[0], 'test')
# plot_magnetic_field_map(stray_plot[0], 'test')
# plotmag = toNumpy(TORCH_MAG)
# plot_magnetic_field_map(plotmag[0], 'test')

