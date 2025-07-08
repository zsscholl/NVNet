import torch
import torch.nn as nn
import numpy as np
from .config import *
from .utils import *
import glob, os, re

# RAW = np.load(DATA_CONFIG['PATH'])
# TORCH_DATA = torch.from_numpy(RAW).transpose(0, 1).to(device=REC_CONFIG['DEVICE'])

gwyd = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\gwys.npy')
TORCH_DATA = torch.from_numpy(gwyd).unsqueeze(0).unsqueeze(0).to(device=REC_CONFIG['DEVICE'])

# MAG_RAW = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\neel_mag1.npy')*600e3*2e-9
# MAG_RAW[2, :, :, :] = np.zeros_like(MAG_RAW[2, :, :, :])
# TORCH_MAG = torch.from_numpy(MAG_RAW).transpose(0, 1).to(device=REC_CONFIG['DEVICE'])
