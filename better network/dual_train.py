import torch
import torch.nn as nn
from backend.config import *
from backend.utils import *
from backend.model import *
from backend.branch_model import *
from backend.data_initializer import *
from backend.forward_transformation import *
import torchmetrics
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from backend.masking import *

matplotlib.use('TkAgg')
MASK = torch_smooth_mask.to(device=REC_CONFIG['DEVICE'], dtype=torch.float32)[:, :, 100:155, 100:155]
MASK = nn.ReflectionPad2d((8, 0, 8, 0))(MASK)

class SingleTrain():
    def __init__(self, dataset, epochs, is_nv=False, display_live_graphs=False):
        super().__init__()
        self.model_x = UBranch().to(REC_CONFIG['DEVICE'])
        self.loss_fn_x = REC_CONFIG['ML_PARAMS']['LOSS_FUNCTION']
        self.losses_x = dict()
        self.optimizer_x = REC_CONFIG['ML_PARAMS']['OPTIMIZER'](self.model_x.parameters())
        self.scheduler_x = REC_CONFIG['ML_PARAMS']['SCHEDULER'](self.optimizer_x)

        shape = dataset.shape[-1]
        shape_ceil = int(2**math.ceil(np.log2(shape)))
        gap = shape_ceil - shape
        slice_start, slice_end = gap//2, gap//2+shape
        self.data = nn.ReflectionPad2d(gap//2)(dataset)
        if self.data.shape[-1] != shape_ceil:
            self.data = nn.ReflectionPad2d((1, 0, 1, 0))(self.data)

        self.xyz = None
        self.propagator = ForwardTransform(shape_ceil).to(device=REC_CONFIG['DEVICE'])

        if is_nv is False:
            self.xyz = self.data
        else:
            self.xyz = self.propagator.NVtoStray(self.data).to(device=REC_CONFIG['DEVICE'])

        self.x = self.xyz[:, 0, :, :].unsqueeze(0)
        graph = PlotML(self.x)

        for epoch in tqdm(range(epochs)):
            self.optimizer_x.zero_grad()
            pred_x = self.model_x(self.x, mask=MASK)
            x_vec = torch.cat([pred_x, torch.zeros_like(pred_x), torch.zeros_like(pred_x)], dim=1)
            output_x = self.propagator.StrayFromMag(x_vec)[:, 0, :, :].unsqueeze(0)
            x_scale = pred_x.std()/output_x.std()
            output_x = output_x*x_scale
            loss_x = self.loss_fn_x(self.x, output_x)

            self.losses_x.update({epoch: loss_x.item()})

            loss_x.backward()

            self.optimizer_x.step()

            self.scheduler_x.step(loss_x)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, X Loss = {self.losses_x[epoch]}')
                if display_live_graphs is True:
                    graph.Render(pred_x, output_x)

test = SingleTrain(TORCH_DATA[:, :, 100:155, 100:155], 5000, is_nv=False, display_live_graphs=True)
# nv_test = ProcessNV(TORCH_IMRE, 5000, is_nv=True, display_live_graphs=True)
