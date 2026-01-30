from NRTA.backend.packages import *
from NRTA.backend.utils import *

class load_data():
    def __init__(self, tensor, guess, mask):
        self.CONFIG = dict()
        self.CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.CONFIG['SHAPE'] = tensor.shape[-1]
        self.CONFIG['DX'] = 2e-9
        self.CONFIG['DY'] = 2e-9
        self.CONFIG['NV_THETA'] = np.deg2rad(54.7)
        self.CONFIG['NV_PHI'] = np.deg2rad(0)
        self.CONFIG['STANDOFF'] = 80e-9
        self.CONFIG['K_MIN'] = 1e-5
        self.CONFIG['K_MAX'] = 1e12
        self.CONFIG['M_AMP'] = 1
        self.CONFIG['LIFETIME'] = 500
        self.CONFIG['MIN_DECAY'] = 0.1
        self.CONFIG['DEPTH'] = 1
        self.CONFIG['L1_WEIGHT'] = 1
        self.CONFIG['TV_WEIGHT'] = 1
        self.data = tensor.to(device=self.CONFIG['DEVICE'], dtype=torch.float32)
        self.guess = guess.to(device=self.CONFIG['DEVICE'], dtype=torch.float32)
        self.mask = mask.to(device=self.CONFIG['DEVICE'], dtype=torch.float32)