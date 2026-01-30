import matplotlib.pyplot as plt

from dipNV.backend.packages import *
from dipNV.backend.utils import *
from dipNV.backend.config import *
from dipNV.masking.mask_maker import dipole_source_mask, dipole_nv_mask
from dipNV.train import *

dip_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\many_dipole.npy')
dipTensor = toTorch(dip_raw)
sqrDipTensor = nn.functional.interpolate(dipTensor, (1810, 800), mode='bilinear', align_corners=True)
dipoleROI = sqrDipTensor[:, :, 100:400, 150:700]
dipoleROI = nn.ConstantPad2d((125, 125, 250, 250), 0)(dipoleROI)
extended_data = nn.functional.interpolate(dipoleROI, (512, 512), mode='bilinear', align_corners=True).detach()
extended_data = torch.where(dipole_nv_mask != 0, extended_data, torch.zeros_like(extended_data))
extended_data[:, :, :, 260:512] = -extended_data[:, :, :, 260:512]
extended_data = extended_data + 0.0001*torch.randn_like(extended_data)
original_nv = toNumpy(extended_data)
# plt.imshow(original_nv, cmap='bwr')
# plt.colorbar()
# plt.show()
# plt.imsave('dipoleNVraw.png', toNumpy(extended_data), cmap='bwr', vmin=-np.max(np.abs(toNumpy(extended_data))), vmax=np.max(np.abs(toNumpy(extended_data))))
# fig, ax = plt.subplots()
# im = ax.imshow(np.abs(1000*toNumpy(extended_data)), cmap='viridis')
# cbar = plt.colorbar(im)
# cbar.set_label(r'$B_{NV}$ (mT)')
# sbar = sbar(1.124, 'nm', length_fraction=0.25)
# ax.add_artist(sbar)
# plt.savefig('dipoleNV_gray.png', dpi=300)
# plt.show()

magnetization = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\dipole_state.npy')[:, 5, :, :]

DIPOLE = dataLoader(extended_data)

# with open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\config_dicts\CONFIG_090125_clov_rot0.json') as json_file:
#     DIPOLE.CONFIG = json.load(json_file)

DIPOLE.CONFIG['DX'] = ((1.4054935523884991e-06)/800)*(512/800)
DIPOLE.CONFIG['K_MIN'] = 1e-8 #2*torch.pi/(1e-6)
DIPOLE.CONFIG['NV']['PHI'] = np.deg2rad(54.75)
DIPOLE.CONFIG['NV']['THETA'] = np.deg2rad(0)
DIPOLE.CONFIG['ML']['L2'] = 1
DIPOLE.CONFIG['ML']['DEPTH'] = 1
DIPOLE.CONFIG['ML']['INIT_LR'] = 0.0001
DIPOLE.CONFIG['MAT_PARAMS']['THICKNESS'] = 25e-9
DIPOLE.CONFIG['ML']['EPOCHS'] = 5000
DIPOLE.CONFIG['ML']['DISPLAY_RATE'] = None
DIPOLE.CONFIG['ML']['SAVE_NV'] = False
DIPOLE.CONFIG['ML']['DO_CLAMPED_RELU'] = False
DIPOLE.CONFIG['K_CUTOFF'] = 0.5*2*torch.pi/(50e-9)
DIPOLE.CONFIG['TEST_K_CUTOFF'] = 2*torch.pi/(1e-6)

sim_mag = toTorch(magnetization).squeeze(0)
sim_mag = torch.where(sim_mag != 0, torch.ones_like(sim_mag), torch.zeros_like(sim_mag))
sim_mag = transform_mask(sim_mag, 2, 0)
sim_mag[:, 0:1, :, :] = -sim_mag[:, 0:1, :, :]
sim_mag[:, 1:, :, :] = torch.zeros_like(sim_mag[:, 1:, :, :])
sim_mag = tv.transforms.GaussianBlur(91, 40)(sim_mag)
DIPOLE.source_mask = sim_mag.to(DIPOLE.device)
for i in range(1, 6):
    standoff = i*25
    DIPOLE.CONFIG['NV']['STANDOFF'] = standoff*1e-9
    DIPOLE.CONFIG['SAVE_NAME'] = f'102325_dip_rot0_standoff{standoff}'
    DIPOLE.CONFIG['K_CUTOFF'] = 0.5*2*torch.pi/(standoff*1e-9)
    test = TrainDIP(DIPOLE)

# fm = forwardModel(DIPOLE)
# analyticMag = fm.analyticReconstruction()
# npMag = toNumpy(analyticMag)
# plt.imshow(npMag[0], cmap='bwr', vmin=-np.max(np.abs(npMag[0])), vmax=np.max(np.abs(npMag[0])))
# plt.colorbar()
# plt.show()
# model = NVNet(DIPOLE.CONFIG['ML']['DEPTH'], False).to(DIPOLE.device)
# model.load_state_dict(torch.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\models\102225_dip_rot0.pth'))
# EvalDIP(model, DIPOLE)

# simulated_mag = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\clover_state.npy')[:, 2, :, :]
# simulated_stray = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\clover_stray_field.npy')[:, 22, :, :]
# projection = np.cos(0)*np.sin(54.7)*simulated_stray[0,:,:] + np.sin(0)*np.sin(54.7)*simulated_stray[1,:,:]+np.cos(54.7)*simulated_stray[2,:,:]
# test = dataLoader(toTorch(projection))
# test.CONFIG['DX'] = 4e-9
# test.CONFIG['NV']['STANDOFF'] = 50e-9
# test.CONFIG['ML']['L1'] = 1
# test.source_mask = None
# hmm = TrainDIP(test)

mx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mx_102225_dip_rot0.npy')
plt.imshow(mx, cmap='bwr', vmin=-np.max(np.abs(mx)), vmax=np.max(np.abs(mx)))
plt.colorbar(label='A/m')
plt.title(r'$M_x$', fontsize=20)
plt.savefig('mx_rot0_dip.png')
plt.show()