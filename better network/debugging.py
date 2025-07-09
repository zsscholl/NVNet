import torch
import matplotlib.pyplot as plt
from backend.forward_transformation import *
from backend.data_initializer import *
from backend.model import *
from backend.utils import *
from train import *
import scipy as scp

transform = ForwardTransform(300)
stray_field = transform.NVtoStray(TORCH_IMRE)
np_stray = toNumpy(stray_field)
rebuild = stray_field[:, 0, :, :]*transform.nv_x + stray_field[:, 1, :, :]*transform.nv_y + stray_field[:, 2, :, :]*transform.nv_z
rebuild = rebuild.unsqueeze(0)

# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(np_stray[0], cmap='bwr')
# ax[1].imshow(np_stray[1], cmap='bwr')
# ax[2].imshow(np_stray[2], cmap='bwr')
# plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(toNumpy(TORCH_IMRE)[120:200, 100:180])
ax[1].imshow(toNumpy(rebuild)[120:200, 100:180])
plt.show()
