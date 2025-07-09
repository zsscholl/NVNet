import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as scp
import torch
import torchvision as tv

rgb_img = cv2.imread(r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\rgb_mask.png')
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)/256
torch_mask = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
# print(torch_mask.shape)
torch_smooth_mask = tv.transforms.functional.gaussian_blur(torch_mask, 3)