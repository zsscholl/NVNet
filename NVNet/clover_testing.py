from NVNet.backend.packages import *
from NVNet.train import *
from NVNet.vector_train import *
from NVNet.backend.vector_model import *

data = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\double_clover.npy')
CLOVER = toTorch(data).to(device=REC_CONFIG['DEVICE'])
test_interpol = nn.functional.interpolate(CLOVER, size=(800, 1600))
TRIMMED = test_interpol[:, :, 190:790, 300:900]
# print(CLOVER.shape)
training = VectorTrain(TRIMMED, torch.zeros_like(CLOVER), epochs=50000, is_nv=True, save_figs=False)
