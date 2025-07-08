import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .config import *

class PadStage(nn.Module):
    def __init__(self):
        super().__init__()
        if DATA_CONFIG['SHAPE'] <= 128:
            self.gap = 128 - DATA_CONFIG['SHAPE']
        elif 128 < DATA_CONFIG['SHAPE'] <= 256:
            self.gap = 256 - DATA_CONFIG['SHAPE']
        elif 256 < DATA_CONFIG['SHAPE'] <= 512:
            self.gap = 512 - DATA_CONFIG['SHAPE']
        else:
            print('size error')

    def forward(self, x):
        return nn.ReflectionPad2d(self.gap // 2)(x)

def toNumpy(tensor):
    return np.squeeze(tensor.cpu().detach().numpy())

def toTorch(matrix):
    return torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)

def plot_magnetic_field_map(B_map, title, cmap='bwr'):
    plt.figure(figsize=(8, 6))
    plt.imshow(B_map, cmap=cmap, aspect='auto')
    plt.colorbar(label="units")
    plt.title(title)
    plt.xlabel("Scan Pixel (X)")
    plt.ylabel("Scan Line (Y)")
    plt.show()