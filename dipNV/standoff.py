from dipNV.backend.packages import *

clov_25 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot0_25nm.json')
clov_50 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot0_50nm.json')
clov_75 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot0_75nm.json')
clov_100 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot0_100nm.json')
dip_25 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_dip_rot0_standoff25nm.json')
dip_50 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_dip_rot0_standoff50nm.json')
dip_75 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_dip_rot0_standoff75nm.json')
dip_100 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_dip_rot0_standoff100nm.json')

clov_25_QOI = json.load(clov_25)
clov_50_QOI = json.load(clov_50)
clov_75_QOI = json.load(clov_75)
clov_100_QOI = json.load(clov_100)
dip_25_QOI = json.load(dip_25)
dip_50_QOI = json.load(dip_50)
dip_75_QOI = json.load(dip_75)
dip_100_QOI = json.load(dip_100)

clov_25_MSE_err = 1e9*np.asarray(list(clov_25_QOI['MSE_LOSS'].values()))
clov_50_MSE_err = 1e9*np.asarray(list(clov_50_QOI['MSE_LOSS'].values()))
clov_75_MSE_err = 1e9*np.asarray(list(clov_75_QOI['MSE_LOSS'].values()))
clov_100_MSE_err = 1e9*np.asarray(list(clov_100_QOI['MSE_LOSS'].values()))
dip_25_MSE_err = 1e9*np.asarray(list(dip_25_QOI['MSE_LOSS'].values()))
dip_50_MSE_err = 1e9*np.asarray(list(dip_50_QOI['MSE_LOSS'].values()))
dip_75_MSE_err = 1e9*np.asarray(list(dip_75_QOI['MSE_LOSS'].values()))
dip_100_MSE_err = 1e9*np.asarray(list(dip_100_QOI['MSE_LOSS'].values()))

standoff = [25, 50, 75, 100]
clov_MSE = [clov_25_MSE_err[-1], clov_50_MSE_err[-1], clov_75_MSE_err[-1], clov_100_MSE_err[-1]]
dip_MSE = [dip_25_MSE_err[-1], dip_50_MSE_err[-1], dip_75_MSE_err[-1], dip_100_MSE_err[-1]]
print(dip_MSE)
print(clov_MSE)
plt.scatter(standoff, clov_MSE, color='red', label='Landau domain')
plt.scatter(standoff, dip_MSE, color='blue', label='Dipole domain')
plt.xlabel('Standoff (nm)', fontsize=12)
plt.ylabel('MSE Error (nT$^2$)', fontsize=12)
plt.locator_params(axis='x', nbins=5)
# plt.legend()
plt.savefig('standoff_curves.png', dpi=300)
plt.show()
