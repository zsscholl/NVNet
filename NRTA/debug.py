from NRTA.backend.packages import *
from NRTA.backend.config import *
# from NRTA.backend.propagation import *
from NRTA.backend.masking import *
from NRTA.dumb_train import *
from NRTA.backend.utils import *

# IMAGE LOADING AND PRE-PROCESSING
dip_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\DIPOLE\many_dipole.npy')
dipTensor = toTorch(dip_raw)
sqrDipTensor = nn.functional.interpolate(dipTensor, (1810, 800), mode='bilinear', align_corners=True)
dipoleROI = sqrDipTensor[:, :, 100:400, 150:700]
dipoleROI = nn.ConstantPad2d((125, 125, 250, 250), 0)(dipoleROI)
_, _, dipx, dipy = dipoleROI.shape
ext_mask = torch.zeros_like(dipoleROI)
ext_mask[:, :, 250:550, 125:675] = 1.0
ext_noise = 0.00023*torch.randn_like(ext_mask).abs()
extended_data = nv_sign_tensor*dipoleROI + ext_noise*(1-ext_mask)
extended_data = tv.transforms.GaussianBlur(3, 1.5)(extended_data)
extended_data = nn.functional.interpolate(extended_data, (512, 512), mode='bilinear', align_corners=True).detach()
DIP_MASK = nn.functional.interpolate(DIP_MASK, (512, 512), mode='bilinear', align_corners=True)
DIP_GUESS = nn.functional.interpolate(DIP_GUESS, (512, 512), mode='bilinear', align_corners=True)
plt.imshow(toNumpy(extended_data), cmap='bwr')
plt.colorbar()
plt.show()

DIPOLE = load_data(extended_data, torch.zeros_like(extended_data), DIP_MASK)
DIPOLE.CONFIG['DX'] = (3.1775607786560826e-06)/800
DIPOLE.CONFIG['DY'] = (3.177560786560826e-06)/800
DIPOLE.CONFIG['NV_THETA'] = np.deg2rad(0)
DIPOLE.CONFIG['NV_PHI'] = np.deg2rad(54.7)
DIPOLE.CONFIG['STANDOFF'] = 50e-9
DIPOLE.CONFIG['LIFETIME'] = 1000
DIPOLE.CONFIG['M_AMP'] = 0.0025
DIPOLE.CONFIG['DEPTH'] = 2
DIPOLE.CONFIG['L1_WEIGHT'] = 1000
DIPOLE.CONFIG['TV_WEIGHT'] = 0.005

# transform = propagator(DIPOLE)
# strayfield = transform.deproject_nv(DIPOLE.data)
# plot_data = toNumpy(strayfield)
# fig, ax = plt.subplots(3)
# ax[0].imshow(plot_data[0], cmap='bwr')
# ax[1].imshow(plot_data[1], cmap='bwr')
# ax[2].imshow(plot_data[2], cmap='bwr')
# plt.show()

testing = TrainNV(DIPOLE, 50000, do_quiver=False)