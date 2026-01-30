import torch

from dipNV.backend.packages import *
from dipNV.backend.utils import *

def png_to_mask(filepath):
    png = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/256
    return toTorch(png)

# DIPOLE MASKING
dipole_nv_mask = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\dipoleNVraw_CLEAN.png')
dipole_nv_mask = nn.functional.interpolate(dipole_nv_mask, (512, 512), mode='nearest')
dipole_nv_mask = tv.transforms.GaussianBlur(3, 1.5)(dipole_nv_mask)

dipole_source_mask = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\masks\dipole_mask1.png')
dipole_source_mask = nn.functional.interpolate(dipole_source_mask, (512, 512), mode='nearest')
dipole_source_mask = torch.cat([dipole_source_mask, torch.zeros_like(dipole_source_mask), torch.zeros_like(dipole_source_mask)], dim=1)
dipole_source_mask = tv.transforms.GaussianBlur(71, 100)(dipole_source_mask)
dipole_source_mask = torch.where(dipole_source_mask != 0, torch.ones_like(dipole_source_mask), dipole_source_mask)

# LANDAU NV MASKING
BLtoTL = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\masks\CLOVER\clover_nvmask_BLtoTR.png')
TLtoBR = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\masks\CLOVER\clover_nvmask_TLtoBR.png')
clover_nvmask = BLtoTL - TLtoBR

# LANDAU MAGNETIZATION MASKING
clover_xmask = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\masks\CLOVER\clover_sourcemask_template.png')
clover_xmask[:, :, :, 264:] = -clover_xmask[:, :, :, 264:]
clover_xmask = tv.transforms.GaussianBlur(41, 70)(clover_xmask)

clover_ymask = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\masks\CLOVER\clover_sourcemask_template.png')
clover_ymask[:, :, 255:, :] = -clover_ymask[:, :, 255:, :]
clover_ymask = tv.transforms.GaussianBlur(41, 70)(clover_ymask)

# clover_zmask = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\masking\masks\CLOVER\clover_coremask.png')
# clover_zmask = tv.transforms.GaussianBlur(21, 40)(clover_zmask)

clover_sourcemask = torch.cat([clover_ymask, clover_xmask, torch.zeros_like(clover_xmask)], dim=1)