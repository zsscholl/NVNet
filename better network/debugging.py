import torch
import matplotlib.pyplot as plt
from backend.forward_transformation import *
from backend.data_initializer import *
from backend.model import *
from backend.utils import *
from train import *
import scipy as scp

# data = TORCH_DATA[:, :, 100:155, 100:155]
transform = ForwardTransform(256)
MAG_X = TORCH_MAG[:, 0, :, :]
MAG_Y = TORCH_MAG[:, 1, :, :]
NEW_MAG_X = torch.cat([MAG_X, torch.zeros_like(MAG_X), torch.zeros_like(MAG_X)], dim=0).unsqueeze(0)
NEW_MAG_Y = torch.cat([torch.zeros_like(MAG_Y), MAG_Y, torch.zeros_like(MAG_Y)], dim=0).unsqueeze(0)
rebuild_x = transform.StrayFromMag(NEW_MAG_X)
rebuild_y = transform.StrayFromMag(NEW_MAG_Y)
plot_data_x = toNumpy(rebuild_x)
plot_data_y = toNumpy(rebuild_y)
plot_magnetic_field_map(plot_data_x[0], 'x')
plot_magnetic_field_map(plot_data_y[1], 'y')
