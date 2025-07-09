import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .config import *

class LivePlot():
    def __init__(self, raw, slice_start, slice_end):
        super().__init__()
        self.fig, self.ax = plt.subplots(3, 3)
        self.data = raw
        self.start = slice_start
        self.end = slice_end
        self.im_raw_x = self.ax[0, 0].imshow(
            self.data[0, 0, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_raw_y = self.ax[0, 1].imshow(
            self.data[0, 1, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_ray_z = self.ax[0, 2].imshow(
            self.data[0, 2, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_mag_x = self.ax[1, 0].imshow(
            self.data[0, 0, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_mag_y = self.ax[1, 1].imshow(
            self.data[0, 1, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_mag_z = self.ax[1, 2].imshow(
            self.data[0, 2, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_rec_x = self.ax[2, 0].imshow(
            self.data[0, 0, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_rec_y = self.ax[2, 1].imshow(
            self.data[0, 1, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')
        self.im_rec_z = self.ax[2, 2].imshow(
            self.data[0, 2, slice_start:slice_end, slice_start:slice_end].cpu().detach().numpy(), cmap='bwr')

    def Render(self, prediction, feedback):
        super().__init__()
        self.im_mag_x.set_data(prediction[0, 0, self.start:self.end, self.start:self.end].cpu().detach().numpy())
        self.im_mag_y.set_data(prediction[0, 1, self.start:self.end, self.start:self.end].cpu().detach().numpy())
        self.im_mag_z.set_data(prediction[0, 2, self.start:self.end, self.start:self.end].cpu().detach().numpy())
        self.im_rec_x.set_data(feedback[0, 0, self.start:self.end, self.start:self.end].cpu().detach().numpy())
        self.im_rec_y.set_data(feedback[0, 1, self.start:self.end, self.start:self.end].cpu().detach().numpy())
        self.im_rec_z.set_data(feedback[0, 2, self.start:self.end, self.start:self.end].cpu().detach().numpy())
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