import torch

from NRTA.backend.packages import *
from NRTA.backend.utils import *

# FOR THE LANDAU
png_x1 = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\long_diag_l.png', cv2.IMREAD_GRAYSCALE)/256
png_x2 = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\long_diag_r.png', cv2.IMREAD_GRAYSCALE)/256
png_y1 = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\short_diag_l.png', cv2.IMREAD_GRAYSCALE)/256
png_y2 = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\short_diag_r.png', cv2.IMREAD_GRAYSCALE)/256
core = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\core.png', cv2.IMREAD_GRAYSCALE)/256

X_GUESS = 0.4*8.6e5*(toTorch(png_x1) - toTorch(png_x2))
Y_GUESS = 0.4*8.6e5*(toTorch(png_y2) - toTorch(png_y1))
Z_GUESS = 0.6*8.6e5*toTorch(core)

GUESS = torch.cat([X_GUESS, Y_GUESS, torch.zeros_like(X_GUESS)], dim=1)
MASK = torch.where(X_GUESS.abs()+Y_GUESS.abs() != 0, torch.ones_like(X_GUESS), torch.zeros_like(X_GUESS))
MASK = torch.cat([MASK, MASK, toTorch(core)], dim=1)
MASK = tv.transforms.GaussianBlur(51, 70)(MASK)

# FOR THE DIPOLE
png_dip = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\dipole_mask1.png', cv2.IMREAD_GRAYSCALE)/256
DIP_MASK = tv.transforms.GaussianBlur(71, 70)(toTorch(png_dip))

DIP_MASK = torch.cat([DIP_MASK, torch.zeros_like(DIP_MASK), torch.zeros_like(DIP_MASK)], dim=1)
DIP_GUESS = -1e-9*8.6e5*DIP_MASK

nv_sign_tensor = -png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\nv_sign_mask.png')
nv_sign_tensor = torch.where(nv_sign_tensor==0, torch.ones_like(nv_sign_tensor), nv_sign_tensor)
nv_sign_tensor = tv.transforms.GaussianBlur(51, 100)(nv_sign_tensor)

landau_input_mask = -png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\landau_nv1.png')
landau_input_mask = torch.where(landau_input_mask==0, torch.ones_like(landau_input_mask), landau_input_mask)
landau_input_mask = tv.transforms.GaussianBlur(51, 100)(landau_input_mask)

mag_horiz = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\horizontal_hourglass.png')
mag_vert = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\vertical_hourglass.png')
input_mask_1 = mag_horiz-mag_vert
input_mask_2 = mag_vert-mag_horiz
magmask = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\landau_magmask.png')
magmask = tv.transforms.GaussianBlur(51, 100)(magmask)
magmask = torch.cat([magmask, magmask, torch.zeros_like(magmask)], dim=1)

guess_horiz = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\guess_horiz_hourglass.png')
guess_vert = png_to_mask(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\guess_vert_hourglass.png')
fac = 2*150e-9*8.6e5
guess_1 = -torch.cat([fac*(guess_horiz-guess_vert), fac*(guess_vert-guess_horiz), torch.zeros_like(guess_horiz)], dim=1)
guess_1 = tv.transforms.GaussianBlur(51, 100)(guess_1)
# plt.imshow(toNumpy(guess_1)[0], cmap='bwr')
# plt.colorbar()
# plt.show()

