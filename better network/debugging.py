import torch
import matplotlib.pyplot as plt
from backend.forward_transformation import *
from backend.data_initializer import *
from backend.model import *
from backend.utils import *

bob = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\gwys.npy')
bob = torch.from_numpy(bob).unsqueeze(0).unsqueeze(0).to(device=REC_CONFIG['DEVICE'])

padding = PadStage()
bob2 = padding(bob)

transform = ForwardTransform(512)
stray = transform.NVtoStray(bob2)
plt.imshow(stray.cpu().numpy()[0, 0], cmap='bwr')
plt.show()