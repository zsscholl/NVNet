from NVNet.backend.packages import *

DATA_CONFIG = dict()
DATA_CONFIG['PATH'] = r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\neel_stray1.npy' #input as a string
DATA_CONFIG['SHAPE'] = 256 #800
DATA_CONFIG['X_PIX_WIDTH'] = 2e-9 #3.478746327915819e-06/800
DATA_CONFIG['Y_PIX_WIDTH'] = 2e-9 #1.7321730629066192e-06/800
DATA_CONFIG['M_SAT'] = 6e5
DATA_CONFIG['THICKNESS'] = 16e-9
DATA_CONFIG['NV_PARAMS'] = dict()
DATA_CONFIG['NV_PARAMS']['THETA'] = np.deg2rad(54.7)
DATA_CONFIG['NV_PARAMS']['PHI'] = np.deg2rad(0)
DATA_CONFIG['NV_PARAMS']['SCAN_HEIGHT'] = 2e-9

REC_CONFIG = dict()
REC_CONFIG['ML_PARAMS'] = dict()
REC_CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
REC_CONFIG['PROP_PARAMS'] = dict()
REC_CONFIG['PROP_PARAMS']['K_CUTOFF'] = 10e12
REC_CONFIG['PROP_PARAMS']['K_EPS'] = 1e-11
REC_CONFIG['PROP_PARAMS']['K_LOW'] = 1e-3
REC_CONFIG['PROP_PARAMS']['K_HIGH'] = 1e5