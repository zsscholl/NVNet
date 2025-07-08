import torch
import torch.nn as nn
import numpy as np

DATA_CONFIG = dict()
DATA_CONFIG['PATH'] = r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\neel_stray1.npy' #input as a string
DATA_CONFIG['SHAPE'] = 256 #300 #only square inputs are allowed
DATA_CONFIG['NV_PARAMS'] = dict()
DATA_CONFIG['NV_PARAMS']['THETA'] = np.deg2rad(54.7)
DATA_CONFIG['NV_PARAMS']['PHI'] = np.deg2rad(135)
DATA_CONFIG['NV_PARAMS']['SCAN_HEIGHT'] = 50e-9

REC_CONFIG = dict()
REC_CONFIG['ML_PARAMS'] = dict()
REC_CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
REC_CONFIG['ML_PARAMS']['LOSS_FUNCTION'] = nn.MSELoss()
REC_CONFIG['ML_PARAMS']['OPTIMIZER'] = lambda params: torch.optim.Adam(params, lr=0.0005)
REC_CONFIG['ML_PARAMS']['SCHEDULER'] = lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode='min',
    factor=0.8,
    patience=200,
    )
REC_CONFIG['PROP_PARAMS'] = dict()
REC_CONFIG['PROP_PARAMS']['K_CUTOFF'] = 10e12
REC_CONFIG['PROP_PARAMS']['K_EPS'] = 1e-3