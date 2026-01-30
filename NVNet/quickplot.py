from NVNet.backend.packages import *

data = np.load(r'C:\Users\zande\PycharmProjects\ANL2025\NVNet\data\double_clover.npy')
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Field Strength (mT)')
plt.title(f'Permalloy')
plt.show()