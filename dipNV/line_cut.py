import matplotlib.pyplot as plt
import numpy as np
from dipNV.masking.mask_maker import clover_nvmask, dipole_nv_mask
from dipNV.backend.packages import *
from dipNV.backend.utils import *

theta = np.deg2rad(54.75)
phi = 0

# LANDAU STATE

raw_clover = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\cropped_clover.npy')
clover_tensor = toTorch(raw_clover)
clover_tensor = torch.where(clover_nvmask != 0, clover_nvmask*clover_tensor, torch.zeros_like(clover_tensor))
clover_tensor = clover_tensor + 0.0001*torch.randn_like(clover_tensor)
original_nv_clov = toNumpy(clover_tensor)

dip_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\many_dipole.npy')
dipTensor = toTorch(dip_raw)
sqrDipTensor = nn.functional.interpolate(dipTensor, (1810, 800), mode='bilinear', align_corners=True)
dipoleROI = sqrDipTensor[:, :, 100:400, 150:700]
dipoleROI = nn.ConstantPad2d((125, 125, 250, 250), 0)(dipoleROI)
extended_data = nn.functional.interpolate(dipoleROI, (512, 512), mode='bilinear', align_corners=True).detach()
extended_data = torch.where(dipole_nv_mask != 0, extended_data, torch.zeros_like(extended_data))
extended_data[:, :, :, 260:512] = -extended_data[:, :, :, 260:512]
extended_data = extended_data + 0.0001*torch.randn_like(extended_data)
original_nv_dip = toNumpy(extended_data)

clov_bx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bx_februn2_clov_rot0_50nm.npy')
clov_amx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMx_februn2_clov_rot0_50nm.npy')
clov_by = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\By_februn2_clov_rot0_50nm.npy')
clov_amy = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMy_februn2_clov_rot0_50nm.npy')
clov_bz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bz_februn2_clov_rot0_50nm.npy')
clov_amz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMz_februn2_clov_rot0_50nm.npy')
clov_mx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mx_februn2_clov_rot0_50nm.npy')
clov_my = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\My_februn2_clov_rot0_50nm.npy')
clov_mz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mz_februn2_clov_rot0_50nm.npy')

clov_b = np.sqrt(clov_bx**2+clov_by**2+clov_bz**2)
clov_m = np.sqrt(clov_mx**2+clov_my**2+clov_mz**2)
clov_am = np.sqrt(clov_amx**2+clov_amy**2+clov_amz**2)
clov_m_nonzero = clov_m.ravel()[np.flatnonzero(clov_m)]
clov_b_reproj = (
    np.cos(theta)*np.sin(phi)*clov_bx +
    np.sin(theta)*np.sin(phi)*clov_by +
    np.cos(phi)*clov_bz
)
clov_am_reproj = (
    np.cos(theta)*np.sin(phi)*clov_amx +
    np.sin(theta)*np.sin(phi)*clov_amy +
    np.cos(phi)*clov_amz
)

plt.imshow(clov_am_reproj)
plt.show()
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bx.png', clov_bx, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\by.png', clov_by, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bz.png', clov_bz, cmap='bwr')

clov_lc_b = np.sum(original_nv_clov[252:260, :], axis=0)/5
clov_lc_am = np.sum(clov_am_reproj[252:260, :], axis=0)/5

clov_std = np.std(np.abs(clov_lc_b) - np.abs(clov_lc_am))
clov_rmse = np.sqrt(np.mean((clov_lc_b - clov_lc_am)**2))

clov_xrange = 2.004*np.arange(0, 512)
plt.scatter(clov_xrange, 1e3*clov_lc_b, c='blue', s=1, label=r'$B_{stray}$')
plt.scatter(clov_xrange, 1e3*clov_lc_am, c='red', s=1, label=r'$AM_{DIP}$')
plt.ylabel(r'Stray Field Magnitude (mT)', fontsize=14)
plt.xlabel(r'Position (nm)', fontsize=14)
plt.legend(loc='best', fontsize=12, markerscale=3)
plt.savefig('clover_linecut_nv.png', dpi=300, bbox_inches='tight')
plt.show()

# DIPOLE

dip_bx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bx_februn2_dip_rot0_standoff50nm.npy')
dip_amx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMx_februn2_dip_rot0_standoff50nm.npy')
dip_by = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\By_februn2_dip_rot0_standoff50nm.npy')
dip_amy = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMy_februn2_dip_rot0_standoff50nm.npy')
dip_bz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bz_februn2_dip_rot0_standoff50nm.npy')
dip_amz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMz_februn2_dip_rot0_standoff50nm.npy')
dip_mx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mx_februn2_dip_rot0_standoff50nm.npy')
dip_my = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\My_februn2_dip_rot0_standoff50nm.npy')
dip_mz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mz_februn2_dip_rot0_standoff50nm.npy')

# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bx.png', dip_bx, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\by.png', dip_by, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bz.png', dip_bz, cmap='bwr')
#
dip_b = np.sqrt(dip_bx**2+dip_by**2+dip_bz**2)
dip_am = np.sqrt(dip_amx**2+dip_amy**2+dip_amz**2)
dip_m = np.sqrt(dip_mx**2+dip_my**2+dip_mz**2)
dip_m = np.where(dip_m <= 1000, np.zeros_like(dip_m), dip_m)
dip_m_nonzero = dip_m.ravel()[np.flatnonzero(dip_m)]

dip_b_reproj = (
    np.cos(theta)*np.sin(phi)*dip_bx +
    np.sin(theta)*np.sin(phi)*dip_by +
    np.cos(phi)*dip_bz
)
dip_am_reproj = (
    np.cos(theta)*np.sin(phi)*dip_amx +
    np.sin(theta)*np.sin(phi)*dip_amy +
    np.cos(phi)*dip_amz
)

dip_lc_b = np.sum(original_nv_dip[252:260, :], axis=0)/5
dip_lc_am = np.sum(dip_am_reproj[252:260, :], axis=0)/5
dip_std = np.std(np.abs(dip_lc_b)-np.abs(dip_lc_am))
dip_rmse = np.sqrt(np.mean((dip_lc_b-dip_lc_am)**2))
dip_xrange = 1.124*np.arange(0, 512)

plt.scatter(dip_xrange, 1e3*dip_lc_b, c='blue', s=1, label=r'$B_{stray}$')
# plt.scatter(dip_xrange, 1e3*dip_lc_am, c='red', s=1, label=r'$AM_{DIP}$')
plt.ylabel(r'Stray Field Magnitude (mT)', fontsize=14)
plt.xlabel(r'Position (nm)', fontsize=14)
plt.legend(loc='best', fontsize=12, markerscale=3)
plt.savefig('dipole_linecut_nv.png', dpi=300, bbox_inches='tight')
plt.show()

# print(np.max(dip_m_nonzero), np.max(clov_m_nonzero))

# plt.imsave('dipole_reproj.png', dip_b_reproj, cmap='viridis', vmin=-np.abs(np.max(dip_b_reproj)), vmax=np.abs(np.max(dip_b_reproj)))
# plt.imsave('landau_reproj.png', clov_b_reproj, cmap='viridis', vmin=-np.abs(np.max(clov_b_reproj)), vmax=np.abs(np.max(clov_b_reproj)))