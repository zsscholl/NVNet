import torch
from dipNV.backend.packages import *
from dipNV.backend.utils import *
from dipNV.backend.config import *
from dipNV.masking.mask_maker import clover_nvmask, clover_sourcemask
# from dipNV.masking.mask_maker import *
from dipNV.train import *
from dipNV.backend.fourier_manager import *

strayfield = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\clover_stray_field.npy')[:, 25, :, :]
# plt.imshow(strayfield[2], cmap='bwr')
# plt.imsave('LANDAU_BZ.png', strayfield[2], cmap='bwr')
# plt.colorbar()
# plt.show()

magnetization = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\clover_ellipse_mag.npy')[:, 5, :, :]
sim_mag = toTorch(magnetization).squeeze(0)
sim_mag = torch.where(sim_mag != 0, torch.ones_like(sim_mag), torch.zeros_like(sim_mag))
sim_mag = transform_mask(sim_mag, 2, 0)

# plt.imshow(toNumpy(mask_tensor)[0], cmap='bwr')
# plt.colorbar()
# plt.show()

# plt.imsave('DIP_MASK_X_0.png', toNumpy(big_mag)[0], cmap='bwr')
# plt.imsave('MASK_Y.png', toNumpy(big_mag)[1], cmap='bwr')
# plt.imsave('MASK_Z.png', toNumpy(big_mag)[2], cmap='bwr')
# plt.imshow(toNumpy(big_mag)[0], cmap='bwr')
# plt.colorbar()
# plt.show()

raw_clover = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\cropped_clover.npy')
clover_tensor = toTorch(raw_clover)
clover_tensor = torch.where(clover_nvmask != 0, clover_nvmask*clover_tensor, torch.zeros_like(clover_tensor))
clover_tensor = clover_tensor + 0.0001*torch.randn_like(clover_tensor)
# clover_tensor = tv.transforms.GaussianBlur(21, 10)(clover_tensor)
original_nv = toNumpy(clover_tensor)
# plt.imshow(original_nv, cmap='bwr')
# plt.colorbar()
# plt.show()
# fig, ax = plt.subplots()
# plt.imsave('CLOVER_MAP.png', toNumpy(clover_tensor), cmap='bwr')
# im = ax.imshow(np.abs(1000*toNumpy(clover_tensor)), cmap='viridis')
# cbar = plt.colorbar(im)
# sbar = sbar(2.004, 'nm', length_fraction=0.25)
# cbar.set_label(r'$B_{NV}$   (mT)')
# ax.add_artist(sbar)
# plt.savefig('cloverNV_gray.png', dpi=300)
# plt.show()

CLOVER = dataLoader(clover_tensor)
CLOVER.CONFIG['DX'] = (1.6030401219940295e-6)/800
CLOVER.CONFIG['K_MIN'] = 2*torch.pi/(1e-6)
CLOVER.CONFIG['NV']['THETA']= np.deg2rad(0)
CLOVER.CONFIG['NV']['PHI'] = np.deg2rad(54.75)
CLOVER.CONFIG['ML']['MSE'] = 1
CLOVER.CONFIG['ML']['DIV'] = 0 #1e-15
CLOVER.CONFIG['ML']['EPOCHS'] = 4000
CLOVER.CONFIG['ML']['DEPTH'] = 1
CLOVER.CONFIG['ML']['INIT_LR'] = 0.00075
CLOVER.CONFIG['ML']['DISPLAY_RATE'] = None
CLOVER.CONFIG['ML']['SAVE_NV'] = False
CLOVER.CONFIG['ML']['DO_CLAMPED_RELU'] = False
CLOVER.CONFIG['MAT_PARAMS']['THICKNESS'] = 25e-9
CLOVER.CONFIG['TEST_K_CUTOFF'] = 0.5*2*torch.pi/(50e-9)
# for z in range(4):
#     for h in range(4):
        # ROT = h*45
ROT = 0
standoff = 50
mag_mask_x = toTorch(create_diagonal_mask(sim_mag.shape[-1], 0))
mag_mask_y = toTorch(create_diagonal_mask(sim_mag.shape[-1], 90))
mag_mask = torch.cat([mag_mask_x, mag_mask_y, torch.zeros_like(mag_mask_y)], dim=1)
mask_tensor = sim_mag * mag_mask
mask_tensor = tv.transforms.GaussianBlur(121, 60)(mask_tensor)
mask_tensor = tv.transforms.functional.rotate(mask_tensor, angle=ROT)
CLOVER.source_mask = mask_tensor.to(CLOVER.device)
# standoff = 25+25*z
CLOVER.CONFIG['NV']['STANDOFF'] = standoff*1e-9
CLOVER.CONFIG['SAVE_NAME'] = f'februn2_clov_rot{ROT}_{standoff}nm'
CLOVER.CONFIG['K_CUTOFF'] = 2 * torch.pi / (standoff*1e-9)
# test = TrainDIP(CLOVER)

model = NVNet(CLOVER.CONFIG['ML']['DEPTH'], False).to(device=CLOVER.device)
model.load_state_dict(torch.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\models\februn2_clov_rot0_50nm.pth', weights_only=True))
EvalDIP(model, CLOVER)
# overseer = overseer(CLOVER)
# stray = overseer.iterative_deprojection(1500, 0.01, 500)
# reproj = overseer.reproject(stray)
# analytic_mag = overseer.iterative_analytic(stray, 1000, 1e3, 500)
# propagated_stray = overseer.propagateMag(analytic_mag)

# plt.imshow(toNumpy(analytic_mag[:, 0:1, :, :]), cmap='bwr')
# plt.colorbar()
# plt.show()
# #
# plt.imshow(toNumpy(stray[:, 1:2, :, :]), cmap='bwr')
# plt.colorbar()
# plt.show()
#
# plt.imshow(toNumpy(stray[:, 2:, :, :]), cmap='bwr')
# plt.colorbar()
# plt.show()

# plt.imshow(original_nv, cmap='bwr')
# plt.colorbar()
# plt.show()

# plt.imshow(toNumpy(reproj), cmap='bwr')
# plt.colorbar()
# plt.show()

# forward_stray = fm.propagateMag(analytic_mag)
# print(nn.MSELoss()(forward_stray, stray)/torch.mean(torch.abs(stray)))

# my = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\My_102225_clov_rot0_quick.npy')
# mx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mx_102225_clov_rot0_quick.npy')
# amx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMx_102225_clov_rot0_quick.npy')
# amy = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMy_102225_clov_rot0_quick.npy')
# amz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMz_102225_clov_rot0_quick.npy')
# bx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bx_102225_clov_rot0_quick.npy')
# by = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\By_102225_clov_rot0_quick.npy')
# bz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bz_102225_clov_rot0_quick.npy')
# plt.imsave('mx_final_clover.png', mx, cmap='bwr', vmin=-np.max(np.abs(mx)), vmax=np.max(np.abs(mx)))
# plt.imsave('my_final_clover.png', my, cmap='bwr', vmin=-np.max(np.abs(my)), vmax=np.max(np.abs(my)))
# plt.imsave('amx_final_clover.png', amx, cmap='bwr', vmin=-np.max(np.abs(amx)), vmax=np.max(np.abs(amx)))
# plt.imsave('amy_final_clover.png', amy, cmap='bwr', vmin=-np.max(np.abs(amy)), vmax=np.max(np.abs(amy)))
# plt.imsave('amz_final_clover.png', amz, cmap='bwr', vmin=-np.max(np.abs(amz)), vmax=np.max(np.abs(amz)))
# plt.imsave('bx_final_clover.png', bx, cmap='bwr', vmin=-np.max(np.abs(bx)), vmax=np.max(np.abs(bx)))
# plt.imsave('by_final_clover.png', by, cmap='bwr', vmin=-np.max(np.abs(by)), vmax=np.max(np.abs(by)))
# plt.imsave('bz_final_clover.png', bz, cmap='bwr', vmin=-np.max(np.abs(bz)), vmax=np.max(np.abs(bz)))

# plt.imshow(mx, cmap='bwr', vmin=-np.max(np.abs(mx)), vmax=np.max(np.abs(mx)))
# plt.colorbar(label='A/m')
# plt.title(r'$M_x$', fontsize=20)
# plt.savefig('mx_rot0.png')
# plt.show()