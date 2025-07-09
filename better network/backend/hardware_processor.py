import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as scp

h5_raw = h5py.File(r'C:\Users\zande\PycharmProjects\ANL2025\better network\data\2025-06-23-15-51-28-odmr_hardware_2mT\seq0\eval\odmr.h5', 'r')
data = np.asarray(h5_raw['data'])
initial_params = [-1e3, 2824000000, 1e6, 1, 1]

spectra = np.array(data['frequency'])
counts = np.array(data['sig'], dtype=np.float64)


def lorentzian(freq, A1, center1, FWHM1, B, C):
    return A1/(1+((freq-center1)/FWHM1)**2)+B*freq+C

fit_params, fit_covar = scp.optimize.curve_fit(lorentzian, spectra[0,0], counts[0,0], initial_params)
# print(fit_params)
plt.scatter(spectra[0,0], counts[0,0], color='red')
plt.plot(spectra[0,0], lorentzian(spectra[0,0], *fit_params), color='blue')
plt.show()