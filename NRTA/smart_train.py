from NRTA.backend.packages import *
from NRTA.backend.model import *
from NRTA.backend.utils import *
from NRTA.backend.propagation import *

class TrainNV():
    def __init__(self, data_dict, epochs, do_quiver):
        self.dev = data_dict.CONFIG['DEVICE']
        noise_tensor = 0.3*0.002*8.6e5*torch.randn_like(data_dict.data).to(device=self.dev)
        self.model = NVNet(depth=1, mask=data_dict.mask, noise=noise_tensor).to(device=self.dev)
        self.L2loss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.losses = dict()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.005)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=50, factor=0.6, threshold=1e-6)
        self.prop = propagator(data_dict)
        b_xyz = self.prop.deproject_nv(data_dict.data)
        graph = PlotML3d(b_xyz)

        for epoch in tqdm(range(epochs)):
            self.optim.zero_grad()
            # guess = 0.5*np.exp(-epoch/data_dict.CONFIG['DECAY_RATE'])*data_dict.initial
            pred = self.model(b_xyz, b_xyz)
            # prop = self.prop.propagate_mag(pred)*150e-9
            loss = 10000*self.L1loss(pred, b_xyz)
            self.losses.update({epoch: loss.item()})
            loss.backward()
            self.optim.step()
            self.sched.step(loss.item())

            if epoch % 200 == 0:
                print(f'Epoch: {epoch}, Loss = {self.losses[epoch]}, LR is {self.sched.get_last_lr()}')
                graph.Render(pred, pred, do_quiver=do_quiver)
