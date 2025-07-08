import torch
import torch.nn as nn
import numpy as np
from backend.config import *
from backend.utils import *
from backend.model import *
from backend.data_initializer import *
from backend.forward_transformation import *
from tqdm import tqdm
import torchmetrics

model = NVNet().to(REC_CONFIG['DEVICE'])
transform = ForwardTransform(512)
padded_raw = PadStage()(TORCH_DATA)*1000
bfield = transform.NVtoStray(padded_raw).to(device=REC_CONFIG['DEVICE'])
plt.imshow(bfield.cpu().detach()[0, 0], cmap='bwr')
plt.colorbar(label="mT")
plt.show()

loss_fn = REC_CONFIG['ML_PARAMS']['LOSS_FUNCTION']
optimizer = REC_CONFIG['ML_PARAMS']['OPTIMIZER'](model.parameters())
scheduler = REC_CONFIG['ML_PARAMS']['SCHEDULER'](optimizer)

model.train()
for epoch in tqdm(range(REC_CONFIG['ML_PARAMS']['EPOCHS'])):
    optimizer.zero_grad()
    prediction = model(bfield)
    feedback = transform.StrayFromMag(prediction)
    loss = loss_fn(prediction, bfield)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss = {loss}')

model.eval()
with torch.no_grad():
    final_output = model(bfield)
    # final_stray = transform.StrayFromMag(final_output)

plot_magnetic_field_map(toNumpy(final_output)[0], 'final out x')
# plot_magnetic_field_map(toNumpy(final_stray)[0][0], 'final stray x')
plot_magnetic_field_map(toNumpy(final_output)[1], 'final out y')
# plot_magnetic_field_map(toNumpy(final_stray)[0][1], 'final stray y')
# plot_magnetic_field_map(toNumpy(final_output)[0][2], 'final out z')
# plot_magnetic_field_map(toNumpy(final_stray)[0][2], 'final stray z')
