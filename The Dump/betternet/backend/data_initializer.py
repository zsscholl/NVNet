import torch
import torch.nn as nn
import numpy as np
from .config import *
from .utils import *
import glob, os, re

# RAW = np.load(DATA_CONFIG['PATH'])
# TORCH_DATA = torch.from_numpy(RAW).transpose(0, 1).to(device=REC_CONFIG['DEVICE'])*1000
# TORCH_DATA[:, 2, :, :] = torch.zeros_like(TORCH_DATA[:, 2, :, :])
#
# MAG_RAW = np.load(r'/betternet\data\neel_mag1.npy')
# TORCH_MAG = torch.from_numpy(MAG_RAW).transpose(0, 1).to(device=REC_CONFIG['DEVICE'])
#
# IMRE_RAW = np.load(r'/betternet\data\gwys.npy')
# TORCH_IMRE = torch.from_numpy(IMRE_RAW).unsqueeze(0).unsqueeze(0).to(device=REC_CONFIG['DEVICE'])*1000