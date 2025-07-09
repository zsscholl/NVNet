import torch
import matplotlib.pyplot as plt
from backend.forward_transformation import *
from backend.data_initializer import *
from backend.model import *
from backend.utils import *
from train import *
import scipy as scp

# debug_data = torch.rand((1, 3, 256, 256)).to(device=REC_CONFIG['DEVICE'])
# test = ProcessNV(debug_data, 5000, is_nv=False, display_graphs=True)

print(TORCH_DATA.shape)
transform = ForwardTransform(256)
propagatedStray = transform.StrayFromMag(TORCH_MAG)
print(propagatedStray.shape)

plotdata = toNumpy(propagatedStray)
plt.imshow(plotdata[0], cmap='bwr')
plt.show()