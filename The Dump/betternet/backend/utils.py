import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .config import *

class MaskedLoss(nn.Module):
    def __init__(self, mask, penalty_weight):
        super().__init__()
        self.mask = mask
        self.penalty_weight = penalty_weight

    def forward(self, pred, target):
        rec_loss = nn.MSELoss()(pred, target)
        disallowed = pred*(1-self.mask)
        penalty = self.penalty_weight * torch.mean(disallowed**2)
        return penalty+rec_loss


class PlotML():
    def __init__(self, input_x, input_y):
        self.fig, self.ax = plt.subplots(2, 3)
        self.stray_x = self.ax[0, 0].imshow(input_x[0,0].cpu().detach().numpy(), cmap='bwr')
        self.mag_x = self.ax[1, 1].imshow(input_x[0,0].cpu().detach().numpy(), cmap='bwr')
        self.rec_x = self.ax[1, 2].imshow(input_x[0,0].cpu().detach().numpy(), cmap='bwr')
        self.stray_y = self.ax[1, 0].imshow(input_y[0,0].cpu().detach().numpy(), cmap='bwr')
        self.mag_y = self.ax[0, 1].imshow(input_y[0,0].cpu().detach().numpy(), cmap='bwr')
        self.rec_y = self.ax[0, 2].imshow(input_y[0,0].cpu().detach().numpy(), cmap='bwr')

    def Render(self, model_y, propagated_y, model_x, propagated_x):
        super().__init__()
        self.mag_x.set_data(model_x[0,0].cpu().detach().numpy())
        self.rec_x.set_data(propagated_x[0,0].cpu().detach().numpy())
        self.mag_y.set_data(model_y[0, 0].cpu().detach().numpy())
        self.rec_y.set_data(propagated_y[0, 0].cpu().detach().numpy())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
class PlotML1d():
    def __init__(self, input):
        self.fig, self.ax = plt.subplots(1, 3)
        self.stray = self.ax[0].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr')
        self.mag = self.ax[1].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr')
        self.rec = self.ax[2].imshow(input[0,0].cpu().detach().numpy(), cmap='bwr')

    def Render(self, model, propagated):
        super().__init__()
        self.mag.set_data(model[0,0].cpu().detach().numpy())
        self.rec.set_data(propagated[0,0].cpu().detach().numpy())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def toNumpy(tensor):
    return np.squeeze(tensor.cpu().detach().numpy())

def toTorch(matrix):
    return torch.from_numpy(matrix).float().unsqueeze(0).unsqueeze(0)

def plot_magnetic_field_map(B_map, title, cmap='bwr'):
    plt.figure(figsize=(8, 6))
    plt.imshow(B_map, cmap=cmap, aspect='auto')
    plt.colorbar(label="units")
    plt.title(title)
    plt.xlabel("Scan Pixel (X)")
    plt.ylabel("Scan Line (Y)")
    plt.show()