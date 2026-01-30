import torch.nn
from dipNV.backend.packages import *
from dipNV.backend.forward_model import *
from dipNV.masking.mask_maker import clover_nvmask, dipole_nv_mask
from dipNV.backend.utils import *
from dipNV.backend.config import *
from dipNV.backend.deprojection import *

raw_clover = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\cropped_clover.npy')
clover_tensor = toTorch(raw_clover)
clover_tensor = torch.where(clover_nvmask != 0, clover_nvmask*clover_tensor, torch.zeros_like(clover_tensor))
clover_tensor = clover_tensor + 0.0001*torch.randn_like(clover_tensor)
# clover_tensor = tv.transforms.GaussianBlur(41, 10)(clover_tensor)
# original_nv = toNumpy(clover_tensor)

CLOVER = dataLoader(clover_tensor)
CLOVER.CONFIG['DX'] = (1.6030401219940295e-6)/800
CLOVER.CONFIG['K_MIN'] = 2*torch.pi/(1e-6)
CLOVER.CONFIG['NV']['THETA']= np.deg2rad(0)
CLOVER.CONFIG['NV']['PHI'] = np.deg2rad(54.75)
CLOVER.CONFIG['ML']['L2'] = 1
CLOVER.CONFIG['ML']['DIV'] = 0 #1e-15
CLOVER.CONFIG['ML']['EPOCHS'] = 2000
CLOVER.CONFIG['ML']['DEPTH'] = 1
CLOVER.CONFIG['ML']['INIT_LR'] = 0.00075
CLOVER.CONFIG['ML']['DISPLAY_RATE'] = 30
CLOVER.CONFIG['ML']['SAVE_NV'] = False
CLOVER.CONFIG['ML']['DO_CLAMPED_RELU'] = False
CLOVER.CONFIG['MAT_PARAMS']['THICKNESS'] = 25e-9
CLOVER.CONFIG['TEST_K_CUTOFF'] = 0.5*2*torch.pi/(50e-9)

# test = deprojector(CLOVER)
# test.iterative_deprojection(epochs=5000, init_lr=0.001, refrate=500)

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

DIPOLE = dataLoader(extended_data)
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

test = deprojector(DIPOLE)
test.iterative_deprojection(epochs=5000, init_lr=0.001, refrate=500)

