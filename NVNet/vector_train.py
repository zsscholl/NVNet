import torch
from NVNet.backend.packages import *
from NVNet.backend.vector_model import *
from NVNet.backend.utils import *

class VectorTrain():

    def __init__(self, dataset, guess, epochs, is_nv=False, save_figs=False):
        super().__init__()
        torch.manual_seed(42)
        shape = dataset.shape[-1]
        shape_ceil = int(2 ** math.ceil(np.log2(shape)))
        gap = shape_ceil - shape
        slice_start, slice_end = gap // 2, gap // 2 + shape
        self.data = nn.ReflectionPad2d(gap // 2)(dataset)
        if self.data.shape[-1] != shape_ceil:
            self.data = nn.ReflectionPad2d((1, 0, 1, 0))(self.data)

        self.model = NV_Net(guess, 0.002, 2, input_masking=False).to(device=REC_CONFIG['DEVICE'])
        self.propagator = ForwardTransform(shape_ceil)
        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.div = diverg().to(device=REC_CONFIG['DEVICE'])
        self.losses = dict()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.007)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100, factor=0.6, threshold=1e-6)

        if is_nv is True:
            self.data = self.propagator.NVtoStray(self.data)[:, :, :, :]
        graph = PlotML3d(self.data)

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            pred = self.model(self.data)
            feedback = self.propagator.StrayFromMag(pred).abs()
            loss = 100*self.L1(feedback, self.data)
            self.losses.update({epoch: loss.item()})
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss.item())

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss = {self.losses[epoch]}, LR is {self.scheduler.get_last_lr()}')
                graph.Render(pred,
                             feedback)
            if epoch % 300 == 0 and save_figs is True:
                graph.Save(epoch)

# data = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\cloverstray.npy')
# TORCH_DATA = torch.from_numpy(data).transpose(0, 1)[:, 0, :, :].unsqueeze(0).to(device=REC_CONFIG['DEVICE'])
