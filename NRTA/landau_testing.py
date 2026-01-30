from NRTA.backend.packages import *
from NRTA.backend.config import *
# from NRTA.backend.propagation import *
from NRTA.backend.masking import *
from NRTA.dumb_train import *
from NRTA.backend.utils import *

raw_npy = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\beautiful.npy')
tensor = toTorch(raw_npy)
tensor = nn.functional.interpolate(tensor, size=(1750, 800), mode='bilinear', align_corners=True)[:, :, 200:690, 200:590]
tensor = nn.ConstantPad2d((61, 61, 11, 11), 0)(tensor)
tensor = tensor*input_mask_1
# plt.imshow(toNumpy(tensor), cmap='bwr')
# plt.colorbar()
# plt.show()

LANDAU = load_data(tensor, torch.zeros_like(tensor), magmask)
LANDAU.CONFIG['DX'] = (1.6030401219940295e-6)/800
LANDAU.CONFIG['DY'] = (1.6030401219940295e-6)/800
LANDAU.CONFIG['NV_THETA'] = np.deg2rad(0)
LANDAU.CONFIG['NV_PHI'] = np.deg2rad(54.7)
LANDAU.CONFIG['STANDOFF'] = 80e-9
LANDAU.CONFIG['LIFETIME'] = 300
LANDAU.CONFIG['M_AMP'] = 0.1*150e-9*8.6e5
LANDAU.CONFIG['MIN_DECAY'] = 0.02
LANDAU.CONFIG['DEPTH'] = 1
LANDAU.CONFIG['L1_WEIGHT'] = 1000
LANDAU.CONFIG['TV_WEIGHT'] = 0

testing = TrainNV(LANDAU, 50000, do_quiver=False)