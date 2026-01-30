from NVNet.backend.packages import *
from NVNet.backend.utils import *
from NVNet.backend.config import *

png = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\skyrmion_mask.png')
grayscale = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)/256
np.save('../../NRTA/data/binary_mask.npy', grayscale)
# mask = nn.functional.interpolate(mask, size=(800, 1600))[:, :, 190:790, 300:900]
# MASK = torch.cat((mask, mask, mask), dim=1)
# MASK = nn.ReflectionPad2d((424, 0, 424, 0))(MASK).to(device=REC_CONFIG['DEVICE'])

# x_pos = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\clover_masks\x_pos.png')
# x_neg = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\clover_masks\x_neg.png')
# y_pos = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\clover_masks\y_pos.png')
# y_neg = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\clover_masks\y_neg.png')
# z = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\clover_masks\z.png')
#
# x_pos_gray = cv2.cvtColor(x_pos, cv2.COLOR_BGR2GRAY)/256
# x_neg_gray = cv2.cvtColor(x_neg, cv2.COLOR_BGR2GRAY)/256
# y_pos_gray = cv2.cvtColor(y_pos, cv2.COLOR_BGR2GRAY)/256
# y_neg_gray = cv2.cvtColor(y_neg, cv2.COLOR_BGR2GRAY)/256
# z_gray = cv2.cvtColor(z,cv2.COLOR_BGR2GRAY)/256
#
# x_comp = x_pos_gray-x_neg_gray
# y_comp = y_pos_gray-y_neg_gray
# z_comp = z_gray
#
# MASK = torch.cat((toTorch(x_comp), toTorch(y_comp), toTorch(z_comp)), dim=1).to(device=REC_CONFIG['DEVICE'])
# MASK = tv.transforms.GaussianBlur(19, 0.3)(MASK)