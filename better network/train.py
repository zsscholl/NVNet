import torch
import torch.nn as nn
from backend.config import *
from backend.utils import *
from backend.model import *
from backend.model2 import *
from backend.data_initializer import *
from backend.forward_transformation import *
import torchmetrics
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from backend.masking import *

# mask_preshape = torch_smooth_mask[:, :, 115:195, 90:170].to(device=REC_CONFIG['DEVICE']).float()
# MASK = nn.functional.pad(mask_preshape, pad = (24,24,24,24), mode='constant', value=0).float()

matplotlib.use('TkAgg') #Necessary for live viewing of model progress (at least in my IDE)
# This model takes the input data and pads it by repeated reflection until its new shape is the closest power of 2
# (i.e. dataset of size 300 is padded until it reaches size 512)
# The model only takes in square datasets
# I've found that this helps with reconstruction artifacts
# If 'is_nv' is set to false, the model accepts a 1x3xLENGTHxLENGTH tensor containing the stray field data for X, Y,
# and Z. This is for when I feed it simulated mumax3 data.
# If 'is_nv' is True, it takes in a 1x1xLENGTHxLENGTH dataset and extracts the stray field according to the NV axis
# specified in the config file.
# 'display_graphs' determines if graphs showing model progress in real time are shown.

class ProcessNV():
    def __init__(self, dataset, epochs, is_nv=False, display_live_graphs=False):
        super().__init__()
        self.model = TwoBranches().to(REC_CONFIG['DEVICE'])
        self.loss_fn = REC_CONFIG['ML_PARAMS']['LOSS_FUNCTION']
        self.losses = dict()
        self.optimizer = REC_CONFIG['ML_PARAMS']['OPTIMIZER'](self.model.parameters())
        self.scheduler = REC_CONFIG['ML_PARAMS']['SCHEDULER'](self.optimizer)

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
            graph = LivePlot(self.data, slice_start, slice_end)
        else:
            self.xyz = self.propagator.NVtoStray(self.data).to(device=REC_CONFIG['DEVICE'])
            graph = LivePlot(self.xyz, slice_start, slice_end)

        self.x = self.xyz[:, 0, :, :].unsqueeze(0)
        self.y = self.xyz[:, 1, :, :].unsqueeze(0)

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            prediction = self.model(self.x, self.y)
            feedback = self.propagator.StrayFromMag(prediction)
            scale_factor = prediction.std()/feedback.std()
            feedback = feedback*scale_factor
            loss = self.loss_fn(self.xyz[:, 0:1, :, :], feedback[:, 0:1, :, :]) #/(1e-18 / 9.27e-24)
#
            self.losses.update({epoch: loss.item()})
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss = {self.losses[epoch]}')
                if display_live_graphs:
                    graph.Render(prediction, feedback)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(self.x, self.y)
            feedback = self.propagator.StrayFromMag(prediction)
        graph.Render(prediction, feedback)
        plt.show()

# test = ProcessNV(TORCH_DATA, 5000, is_nv=False, display_live_graphs=True)
nv_test = ProcessNV(TORCH_IMRE, 5000, is_nv=True, display_live_graphs=True)
