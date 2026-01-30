import matplotlib.pyplot as plt
import numpy as np

from dipNV.backend.packages import *
from dipNV.backend.utils import *

# LANDAU STATE

clov_bx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bx_102225_clov_rot0_quick.npy')
clov_abx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMx_102225_clov_rot0_quick.npy')
clov_by = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\By_102225_clov_rot0_quick.npy')
clov_aby = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMy_102225_clov_rot0_quick.npy')
clov_bz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bz_102225_clov_rot0_quick.npy')
clov_abz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMz_102225_clov_rot0_quick.npy')
clov_mx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mx_102225_clov_rot0_quick.npy')
clov_my = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\My_102225_clov_rot0_quick.npy')
clov_mz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mz_102225_clov_rot0_quick.npy')

clov_b = np.sqrt(clov_bx**2+clov_by**2+clov_bz**2)
clov_m = np.sqrt(clov_mx**2+clov_my**2+clov_mz**2)
clov_ab = np.sqrt(clov_abx**2+clov_aby**2+clov_abz**2)
clov_m_nonzero = clov_m.ravel()[np.flatnonzero(clov_m)]
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bx.png', clov_bx, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\by.png', clov_by, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bz.png', clov_bz, cmap='bwr')

clov_lc_b = np.sum(clov_b[252:257, :], axis=0)/5
clov_lc_ab = np.sum(clov_ab[252:257, :], axis=0)/5

clov_std = np.std(np.abs(clov_lc_b) - np.abs(clov_lc_ab))
clov_rmse = np.sqrt(np.mean((clov_lc_b - clov_lc_ab)**2))
print(clov_rmse)

clov_xrange = 2.004*np.arange(0, 512)
plt.scatter(clov_xrange, 1e3*clov_lc_b, c='blue', s=1, label=r'$B_{stray}$')
plt.scatter(clov_xrange, 1e3*clov_lc_ab, c='red', s=1, label=r'$AM_{DIP}$')
plt.ylabel(r'Stray Field Magnitude (mT)', fontsize=14)
plt.xlabel(r'Position (nm)', fontsize=14)
plt.legend(loc='best', fontsize=12, markerscale=3)
# plt.savefig('good_clover_linecut.png', dpi=300, bbox_inches='tight')
# plt.show()

# DIPOLE

dip_bx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bx_102225_dip_rot0.npy')
dip_abx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMx_102225_dip_rot0.npy')
dip_by = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\By_102225_dip_rot0.npy')
dip_aby = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMy_102225_dip_rot0.npy')
dip_bz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Bz_102225_dip_rot0.npy')
dip_abz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\AMz_102225_dip_rot0.npy')
dip_mx = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mx_102225_dip_rot0.npy')
dip_my = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\My_102225_dip_rot0.npy')
dip_mz = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\arrays\Mz_102225_dip_rot0.npy')

# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bx.png', dip_bx, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\by.png', dip_by, cmap='bwr')
# plt.imsave(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\for_cuts\bz.png', dip_bz, cmap='bwr')

dip_b = np.sqrt(dip_bx**2+dip_by**2+dip_bz**2)
dip_ab = np.sqrt(dip_abx**2+dip_aby**2+dip_abz**2)
dip_m = np.sqrt(dip_mx**2+dip_my**2+dip_mz**2)
dip_m = np.where(dip_m <= 1000, np.zeros_like(dip_m), dip_m)
dip_m_nonzero = dip_m.ravel()[np.flatnonzero(dip_m)]
dip_lc_b = np.sum(dip_b[276:281, :], axis=0)/5
dip_lc_ab = np.sum(dip_ab[276:281, :], axis=0)/5
dip_std = np.std(np.abs(dip_lc_b)-np.abs(dip_lc_ab))
dip_rmse = np.sqrt(np.mean((dip_lc_b-dip_lc_ab)**2))
print(dip_rmse)
dip_xrange = 1.124*np.arange(0, 512)

plt.scatter(dip_xrange, 1e3*dip_lc_b, c='blue', s=1, label=r'$B_{stray}$')
plt.scatter(dip_xrange, 1e3*dip_lc_ab, c='red', s=1, label=r'$AM_{DIP}$')
plt.ylabel(r'Stray Field Magnitude (mT)', fontsize=14)
plt.xlabel(r'Position (nm)', fontsize=14)
plt.legend(loc='best', fontsize=12, markerscale=3)
# plt.savefig('good_dipole_linecut.png', dpi=300, bbox_inches='tight')
# plt.show()