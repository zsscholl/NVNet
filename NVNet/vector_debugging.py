import torch

from NVNet.backend.packages import *
from NVNet.train import *
from NVNet.vector_train import *
from NVNet.backend.vector_model import *

clover_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_stray_field.npy')[0:3, 9, :, :]
from NVNet.vector_train import *
CLOVER_DATA = torch.from_numpy(clover_raw).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
landau_raw = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_state.npy')[:, 7, :, :]
LANDAU_DATA = torch.from_numpy(landau_raw).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
CLOVER_DATA = tv.transforms.GaussianBlur(19, 10)(CLOVER_DATA).abs()
test = VectorTrain(CLOVER_DATA, LANDAU_DATA, 50000, is_nv=False, save_figs=False)

# skyrmion = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\skyrmion_stray.npy')[0:3, 8, :]
# SKYRM_DATA = torch.from_numpy(skyrmion).to(device=REC_CONFIG['DEVICE']).unsqueeze(0)
# skyrmag = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\skyrmion_mag.npy')[0:3, 7, :]
# GUESS = torch.from_numpy(skyrmag).unsqueeze(0).to(device=REC_CONFIG['DEVICE'])
# theta = np.deg2rad(DATA_CONFIG['NV_PARAMS']['THETA'])
# phi = np.deg2rad(DATA_CONFIG['NV_PARAMS']['PHI'])
# test = np.cos(theta)*np.sin(phi)*skyrmion[0,0] + np.sin(theta)*np.sin(phi)*skyrmion[1,0]+np.cos(phi)*skyrmion[2,0]
# data = toTorch(test).to(device=REC_CONFIG['DEVICE'])

# data2 = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\gwys.npy')
# data1 = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\afm_trace.npy')
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(data1, cmap='viridis')
# ax[1].imshow(data2, cmap='bwr')
# plt.savefig('afm_plot', dpi=1200, bbox_inches='tight')
# plt.show()