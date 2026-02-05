from dipNV.backend import *
from dipNV.backend.packages import *
from dipNV.backend.utils import *

class dataLoader():
    def __init__(self, tensor):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nv_data = tensor.to(self.device)
        self.source_mask = None
        self.test_stray  = None
        self.CONFIG = dict()
        self.CONFIG['DX'] = None
        self.CONFIG['MAT_PARAMS'] = dict()
        self.CONFIG['MAT_PARAMS']['M_SAT'] = 8.6e5
        self.CONFIG['MAT_PARAMS']['THICKNESS'] = 20e-9
        self.CONFIG['NV'] = dict()
        self.CONFIG['NV']['THETA'] = np.deg2rad(0)
        self.CONFIG['NV']['PHI'] = np.deg2rad(54.7)
        self.CONFIG['NV']['STANDOFF'] = 50e-9
        self.CONFIG['K_MIN'] = 2*torch.pi/(100e-9)
        self.CONFIG['ML'] = dict()
        self.CONFIG['ML']['DEPTH'] = 1
        self.CONFIG['ML']['EPOCHS'] = 10000
        self.CONFIG['ML']['MSE'] = 1
        self.CONFIG['ML']['INIT_LR'] = 0.001
        self.CONFIG['ML']['DISPLAY_RATE'] = 100
        self.CONFIG['ML']['SAVE_NV'] = False
        self.CONFIG['ML']['DO_CLAMPED_RELU'] = False
        self.CONFIG['SAVE_NAME'] = 'temp'
        self.CONFIG['K_CUTOFF'] = 2*torch.pi/self.CONFIG['NV']['STANDOFF']
        self.CONFIG['TEST_K_CUTOFF'] = 2*torch.pi/self.CONFIG['NV']['STANDOFF']