from dipNV.backend.packages import *
from dipNV.backend.utils import *
from dipNV.backend.config import *
from dipNV.masking.mask_maker import clover_nvmask, clover_sourcemask
# from dipNV.masking.mask_maker import *
from dipNV.train import *

raw_data = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\data\afm.numpy') - 0.002*np.cos(np.deg2rad(54.7))
raw_tensor = toTorch(raw_data)
raw_tensor = nn.functional.interpolate(raw_tensor, size=(512, 512), mode='bilinear', align_corners=True)

afm_mask = torch.where(raw_tensor >= 1e-6, -torch.ones_like(raw_tensor), torch.ones_like(raw_tensor))
afm_mask = tv.transforms.GaussianBlur(17, 5)(afm_mask)
source_mask = torch.cat((torch.zeros_like(afm_mask), torch.zeros_like(afm_mask), afm_mask), dim=1)

AFM = dataLoader(raw_tensor)
AFM.CONFIG['DX'] = 5e-06
AFM.CONFIG['NV']['STANDOFF'] = 50e-9
AFM.CONFIG['NV']['THETA']= np.deg2rad(0)
AFM.CONFIG['NV']['PHI'] = np.deg2rad(54.7)
AFM.CONFIG['ML']['L2'] = 1
AFM.CONFIG['ML']['DIV'] = 0 #1e-15
AFM.CONFIG['ML']['EPOCHS'] = 20000
AFM.CONFIG['ML']['DEPTH'] = 2
AFM.CONFIG['ML']['INIT_LR'] = 0.001
AFM.CONFIG['ML']['DISPLAY_RATE'] = 50
AFM.CONFIG['K_CUTOFF'] = 0.009 * AFM.CONFIG['K_CUTOFF']
AFM.CONFIG['SAVE_NAME'] = 'afm'
AFM.source_mask = source_mask.to(device=AFM.device)

testing = TrainDIP(AFM)

# model = NVNet(CLOVER.CONFIG['ML']['DEPTH']).to(device=CLOVER.device)
# model.load_state_dict(torch.load(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\models\081525_clover_rot180.pth', weights_only=True))
# EvalDIP(model, CLOVER)
# fm = forwardModel(CLOVER)
# stray = fm.deprojectNV()
# analytic_mag = fm.analyticReconstruction()
# forward_stray = fm.propagateMag(analytic_mag)
# print(nn.MSELoss()(forward_stray, stray)/torch.mean(torch.abs(stray)))