import torch
from dipNV.backend.packages import *
from dipNV.backend.utils import *
from dipNV.backend.config import *
from dipNV.masking.mask_maker import clover_nvmask, clover_sourcemask
# from dipNV.masking.mask_maker import *
from dipNV.train import *
from dipNV.backend.fourier_manager import *

phi = np.deg2rad(54.7)
theta = 0

stray = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\clover_ellipse_stray.npy')[:, 15, :, :]
magnetization = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\clover_ellipse_mag.npy')[:, 5, :, :]
nv_input = np.cos(phi)*np.sin(theta)*stray[0, :, :]+np.sin(phi)*np.sin(theta)*stray[1, :, :]+np.cos(phi)*np.cos(theta)*stray[2, :, :]
nv_input = toTorch(nv_input)

MUMAX = dataLoader(nv_input)
MUMAX.CONFIG['DX'] = (1.6030401219940295e-6)/800
MUMAX.CONFIG['K_MIN'] = 2*torch.pi/(1e-6)
MUMAX.CONFIG['NV']['THETA']= np.deg2rad(0)
MUMAX.CONFIG['NV']['PHI'] = np.deg2rad(54.75)
MUMAX.CONFIG['ML']['MSE'] = 1
MUMAX.CONFIG['ML']['DIV'] = 0 #1e-15
MUMAX.CONFIG['ML']['EPOCHS'] = 4000
MUMAX.CONFIG['ML']['DEPTH'] = 1
MUMAX.CONFIG['ML']['INIT_LR'] = 0.00075
MUMAX.CONFIG['ML']['DISPLAY_RATE'] = 30
MUMAX.CONFIG['ML']['SAVE_NV'] = False
MUMAX.CONFIG['ML']['DO_CLAMPED_RELU'] = False
MUMAX.CONFIG['MAT_PARAMS']['THICKNESS'] = 25e-9
MUMAX.CONFIG['TEST_K_CUTOFF'] = 0.5*2*torch.pi/(50e-9)
MUMAX.CONFIG['NV']['STANDOFF'] = 50e-9
MUMAX.CONFIG['SAVE_NAME'] = f'februn3_mumax'
MUMAX.test_stray = toTorch(stray).to(device=MUMAX.device).squeeze(0)
MUMAX.source_mask = toTorch(magnetization).to(device=MUMAX.device).squeeze(0)

# test = TrainDIP(MUMAX)

# model = NVNet(MUMAX.CONFIG['ML']['DEPTH'], False).to(device=MUMAX.device)
# model.load_state_dict(torch.load(f'C:/Users/zande/PycharmProjects/ANL2025/dipNV/output/models/februn3_mumax.pth', weights_only=True))
# EvalDIP(model, MUMAX)

QOI = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_mumax.json'))
epochs = np.asarray(list(QOI['MSE_LOSS'].keys()), dtype=float)
mse_err = 1e6*np.asarray(list(QOI['MSE_LOSS'].values()))
SNR = np.asarray(list(QOI['SNR'].values()))

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.15)  # Adjust space between axes
ax0.scatter(epochs, mse_err, s=1, color='red', label=r'$0\degree$')
ax1.scatter(epochs, mse_err, s=1, color='red', label=r'$0\degree$')
ax0.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
ax0.set_ylim(2,10)
ax1.set_ylim(0, 2)
ax0.spines['bottom'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax0.xaxis.tick_top()
ax0.tick_params(labeltop='off', pad=10)
ax1.tick_params(labeltop='off', pad=10)
ax1.xaxis.tick_bottom()
ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=False))
ax0.tick_params(direction='in', top=True, right=True, bottom=False, labeltop=False)
ax1.tick_params(direction='in', top=False, right=True)
d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax0.plot([0, 1], [0, 0], transform=ax0.transAxes, **kwargs)
ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
fig.supylabel(r'Mean Square Error ($\mu$T$^2$)', fontsize=12)
fig.supxlabel('Epoch', fontsize=12)
plt.savefig('mumax_error_curve.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.show()

print(np.sqrt(mse_err[-1]))