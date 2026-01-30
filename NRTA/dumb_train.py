from NRTA.backend.packages import *
from NRTA.backend.upsample_model import *
from NRTA.backend.utils import *
from NRTA.backend.propagation import *

class TrainNV():
    def __init__(self, data_dict, epochs, do_quiver):
        self.dev = data_dict.CONFIG['DEVICE']
        self.model = NVNet(data_dict).to(device=self.dev)
        self.L2loss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.tv = tm.image.TotalVariation().to(self.dev)
        self.total_losses = dict()
        self.L1_losses = dict()
        self.TV_losses = dict()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=50, factor=0.6, threshold=1e-9)
        self.prop = propagator(data_dict)
        b_xyz = self.prop.deproject_nv(data_dict.data)
        graph = PlotML3d(b_xyz)
        self.noise = 0.05*torch.randn_like(b_xyz)

        for epoch in tqdm(range(epochs)):
            self.optim.zero_grad()
            pred = self.model(b_xyz+self.noise)+torch.clamp(np.exp(-epoch/data_dict.CONFIG['LIFETIME'])*data_dict.guess,
                                                 min=data_dict.CONFIG['MIN_DECAY'])
            prop = self.prop.propagate_mag(pred)
            L1loss = data_dict.CONFIG['L1_WEIGHT']*self.L1loss(prop, b_xyz)
            TVloss = data_dict.CONFIG['TV_WEIGHT']*self.tv(pred)
            loss = L1loss+TVloss
            self.total_losses.update({epoch: loss.item()})
            self.L1_losses.update({epoch: L1loss.item()})
            self.TV_losses.update({epoch: TVloss.item()})
            loss.backward()
            self.optim.step()
            self.sched.step(loss.item())

            if epoch % 200 == 0:
                print(f'Epoch: {epoch}, L1loss = {self.L1_losses[epoch]}, TVloss = {self.TV_losses[epoch]}, LR is {self.sched.get_last_lr()}')
                graph.Render(pred, prop)