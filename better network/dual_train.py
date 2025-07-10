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
MASK = torch_smooth_mask.to(device=REC_CONFIG['DEVICE'], dtype=torch.float32)

class SingleTrain():
    def __init__(self, dataset, epochs, is_nv=False, display_live_graphs=False):
        super().__init__()
        self.model_x = UBranch().to(REC_CONFIG['DEVICE'])
        self.loss_fn_x = REC_CONFIG['ML_PARAMS']['LOSS_FUNCTION']
        self.losses_x = dict()
        self.optimizer_x = REC_CONFIG['ML_PARAMS']['OPTIMIZER'](self.model_x.parameters())
        self.scheduler_x = REC_CONFIG['ML_PARAMS']['SCHEDULER'](self.optimizer_x)

        self.model_y = UBranch().to(REC_CONFIG['DEVICE'])
        self.loss_fn_y = REC_CONFIG['ML_PARAMS']['LOSS_FUNCTION']
        self.losses_y = dict()
        self.optimizer_y = REC_CONFIG['ML_PARAMS']['OPTIMIZER'](self.model_y.parameters())
        self.scheduler_y = REC_CONFIG['ML_PARAMS']['SCHEDULER'](self.optimizer_y)

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
        self.y = self.xyz[:, 1, :, :].unsqueeze(0)
        graph = PlotML(self.x, self.y)

        for epoch in tqdm(range(epochs)):
            self.optimizer_x.zero_grad()
            pred_x = self.model_x(self.x, MASK)
            x_vec = torch.cat([pred_x, torch.zeros_like(pred_x), torch.zeros_like(pred_x)], dim=1)
            output_x = self.propagator.StrayFromMag(x_vec)[:, 0, :, :].unsqueeze(0)
            x_scale = pred_x.std()/output_x.std()
            output_x = output_x*x_scale
            loss_x = self.loss_fn_x(self.x, output_x)
            self.losses_x.update({epoch: loss_x.item()})
            loss_x.backward()
            self.optimizer_x.step()
            self.scheduler_x.step(loss_x)

            self.optimizer_y.zero_grad()
            pred_y = self.model_y(self.y, MASK)
            y_vec = torch.cat([torch.zeros_like(pred_y), pred_y, torch.zeros_like(pred_y)], dim=1)
            output_y = self.propagator.StrayFromMag(y_vec)[:, 1, :, :].unsqueeze(0)
            y_scale = pred_y.std() / output_y.std()
            output_y = output_y * y_scale
            loss_y = self.loss_fn_y(self.y, output_y)
            self.losses_y.update({epoch: loss_y.item()})
            loss_y.backward()
            self.optimizer_y.step()
            self.scheduler_y.step(loss_y)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, X Loss = {self.losses_x[epoch]}')
                if display_live_graphs is True:
                    graph.Render(pred_x, output_x, pred_y, output_y)

test = SingleTrain(TORCH_DATA[:, :, 100:155, 100:155], 5000, is_nv=False, display_live_graphs=True)
# nv_test = ProcessNV(TORCH_IMRE, 5000, is_nv=True, display_live_graphs=True)
