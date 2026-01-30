import torch
from NVNet.backend.packages import *
from NVNet.backend.model import *
from NVNet.backend.utils import *

# MASKING THE MODEL OUTPUT
rgb = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\landau_rgb.png')
grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)/256
MASK = toTorch(grayscale)
MASK = tv.transforms.functional.gaussian_blur(MASK, (3, 3)).to(device=REC_CONFIG['DEVICE'])

class Train():
    def __init__(self, dataset, epochs, is_nv=False):
        super().__init__()
        shape = dataset.shape[-1]
        shape_ceil = int(2 ** math.ceil(np.log2(shape)))
        gap = shape_ceil - shape
        slice_start, slice_end = gap // 2, gap // 2 + shape
        self.data = nn.ReflectionPad2d(gap // 2)(dataset)
        if self.data.shape[-1] != shape_ceil:
            self.data = nn.ReflectionPad2d((1, 0, 1, 0))(self.data)

        self.model = NV_Net(2, nn.LeakyReLU()).to(device=REC_CONFIG['DEVICE'])
        self.propagator = ForwardTransform(shape_ceil)
        self.loss_fn = nn.MSELoss()
        self.losses = dict()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=200, factor=0.8)
        graph = PlotML1d(dataset)

        if is_nv is True:
            self.data = self.propagator.NVtoStray(self.data)[:, 0:1, :, :]

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            pred = self.model(self.data)*MASK
            pred_vec = torch.cat([pred, torch.zeros_like(pred), torch.zeros_like(pred)], dim=1)
            feedback = self.propagator.StrayFromMag(pred_vec)[:, 0:1, :, :]
            loss = self.loss_fn(feedback, self.data)
            self.losses.update({epoch: loss.item()})
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss.item())

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss = {self.losses[epoch]}, LR is {self.scheduler.get_last_lr()}')
                graph.Render(pred,
                             feedback)

# data = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\cloverstray.npy')
# TORCH_DATA = torch.from_numpy(data).transpose(0, 1)[:, 0, :, :].unsqueeze(0).to(device=REC_CONFIG['DEVICE'])
