from NRTA.backend.packages import *
from NRTA.backend.config import *
from NRTA.backend.propagation import *
from NRTA.backend.propagation import *
from NRTA.backend.masking import *
from NRTA.dumb_train import *
from NRTA.backend.utils import *

# PRE-PROCESSING
raw_npy = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\beautiful.npy')
# plt.imshow(raw_npy, cmap='viridis')
# plt.show()
tensor = toTorch(raw_npy)
tensor = nn.functional.interpolate(tensor, size=(1750, 800), mode='bilinear', align_corners=True)[:, :, 200:690, 200:590]
tensor = nn.ConstantPad2d((61, 61, 11, 11), 0)(tensor)
plt.imshow(toNumpy(tensor), cmap='viridis')
np.save('cropped_clover.npy', toNumpy(tensor))
tensor = tv.transforms.GaussianBlur(51, 0.7)(tensor)
# plt.imshow(toNumpy(tensor), cmap='viridis', origin='lower')
# plt.show()

# png_mask = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NRTA\data\masks\mask2.png', cv2.IMREAD_GRAYSCALE)/256
# tensor_mask = toTorch(png_mask)
# tensor_mask = torch.cat([tensor_mask, tensor_mask, tensor_mask], dim=1)
# tensor_mask[:, 2:, :, :] = 0
# tensor_mask[:, 2:, 232:280, 232:280] = 1
# tensor_mask = tv.transforms.GaussianBlur(95, 30)(tensor_mask)
# plt.imshow(toNumpy(tensor_mask)[0], cmap='gray')
# plt.colorbar()
# plt.show()

tensor_guess = torch.zeros((1, 3, 512, 512))
tensor_guess[:, 1:2, 0:250, :] = -6e5
tensor_guess[:, 1:2, 250:, :] = 6e5
tensor_guess[:, 0:1, :, 0:250] = 6e5
tensor_guess[:, 0:1, :, 250:] = -6e5
tensor_guess[:, 2:, :, :] = 6e5

clover_data = load_data(tensor, guess=GUESS, mask=MASK)
clover_data.CONFIG['DX'] = 1.6030401219940295e-6/800
clover_data.CONFIG['DY'] = 1.6030401219940295e-6/800
clover_data.CONFIG['NV_THETA'] = np.deg2rad(0)
clover_data.CONFIG['NV_PHI'] = np.deg2rad(54.7)
clover_data.CONFIG['STANDOFF'] = 50e-9
clover_data.CONFIG['DECAY_RATE'] = 2000

# clover_data.initial = torch.zeros_like(tensor).to(device=clover_data.CONFIG['DEVICE'])

training = TrainNV(clover_data, 50000, do_quiver=False)