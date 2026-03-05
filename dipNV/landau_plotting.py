from dipNV.backend.packages import *

f0 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot0_50nm.json')
f45 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot45_50nm.json')
f90 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot90_50nm.json')
f135 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn2_clov_rot135_50nm.json')

QOI_0 = json.load(f0)
QOI_45 = json.load(f45)
QOI_90 = json.load(f90)
QOI_135 = json.load(f135)
epochs = np.asarray(list(QOI_0['MSE_LOSS'].keys()), dtype=float)
mse_err_0 = 1e6*np.asarray(list(QOI_0['MSE_LOSS'].values()))
mse_err_45 = 1e6*np.asarray(list(QOI_45['MSE_LOSS'].values()))
mse_err_90 = 1e6*np.asarray(list(QOI_90['MSE_LOSS'].values()))
mse_err_135 = 1e6*np.asarray(list(QOI_135['MSE_LOSS'].values()))
SNR_0 = np.asarray(list(QOI_0['SNR'].values()))
SNR_45 = np.asarray(list(QOI_45['SNR'].values()))
SNR_90 = np.asarray(list(QOI_90['SNR'].values()))
SNR_135 = np.asarray(list(QOI_135['SNR'].values()))

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.15)  # Adjust space between axes
ax0.scatter(epochs, mse_err_0, s=1, color='red', label=r'$0\degree$')
ax0.scatter(epochs, mse_err_45, s=1, color='orange', label=r'$45\degree$')
ax0.scatter(epochs, mse_err_90, s=1, color='olive', label=r'$90\degree$')
ax0.scatter(epochs, mse_err_135, s=1, color='olive', label=r'$90\degree$')

ax1.scatter(epochs, mse_err_0, s=1, color='red', label=r'$0\degree$')
ax1.scatter(epochs, mse_err_45, s=1, color='orange', label=r'$45\degree$')
ax1.scatter(epochs, mse_err_90, s=1, color='orange', label=r'$75\degree$')
ax1.scatter(epochs, mse_err_135, s=1, color='olive', label=r'$90\degree$')

ax0.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))

ax0.set_ylim(2,10)
ax1.set_ylim(0, 2)
ax0.spines['bottom'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax0.xaxis.tick_top()
ax0.tick_params(labeltop='off', pad=10)
ax1.tick_params(labeltop='off', pad=10)
ax1.xaxis.tick_bottom()
ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=False))
ax0.tick_params(direction='in', top=True, right=True, bottom=False, labeltop=False)
ax1.tick_params(direction='in', top=False, right=True)
d = .25  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax0.plot([0, 1], [0, 0], transform=ax0.transAxes, **kwargs)
ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
fig.supylabel(r'Mean Square Error ($\mu$T$^2$)', fontsize=12)
fig.supxlabel('Epoch', fontsize=12)
plt.savefig('landau_error_curves.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.show()

snr_array = [SNR_0[-1], SNR_45[-1], SNR_90[-1], SNR_135[-1]]
print(snr_array)
mse_array = [mse_err_0[-1], mse_err_45[-1], mse_err_90[-1], mse_err_135[-1]]
print(np.sqrt(mse_array))
ax_snr_mse = plt.subplot()
ax_snr_mse.scatter(mse_array, snr_array, color='red', label='SNR')
plt.show()

# clov_standoff_25 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_landau_rot0_25nm.json'))
# clov_standoff_50 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_landau_rot0_50nm.json'))
# clov_standoff_75 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_landau_rot0_75nm.json'))
# clov_standoff_100 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_landau_rot0_100nm.json'))
# clov_standoff_125 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_landau_rot0_125nm.json'))
#
# dip_standoff_25 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_090925_dip_standoff25nm.json'))
# dip_standoff_50 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_090925_dip_standoff50nm.json'))
# dip_standoff_75 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_090925_dip_standoff75nm.json'))
# dip_standoff_100 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_090925_dip_standoff100nm.json'))
# dip_standoff_125 = json.load(open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_090925_dip_standoff125nm.json'))
#
# clov_error_25 = 100*np.asarray(list(clov_standoff_25['REL_ERR'].values()))[-1]
# clov_error_50 = 100*np.asarray(list(clov_standoff_50['REL_ERR'].values()))[-1]
# clov_error_75 = 100*np.asarray(list(clov_standoff_75['REL_ERR'].values()))[-1]
# clov_error_100 = 100*np.asarray(list(clov_standoff_100['REL_ERR'].values()))[-1]
# clov_error_125 = 100*np.asarray(list(clov_standoff_125['REL_ERR'].values()))[-1]
#
# dip_error_25 = 100*np.asarray(list(dip_standoff_25['REL_ERR'].values()))[-1]
# dip_error_50 = 100*np.asarray(list(dip_standoff_50['REL_ERR'].values()))[-1]
# dip_error_75 = 100*np.asarray(list(dip_standoff_75['REL_ERR'].values()))[-1]
# dip_error_100 = 100*np.asarray(list(dip_standoff_100['REL_ERR'].values()))[-1]
# dip_error_125 = 100*np.asarray(list(dip_standoff_125['REL_ERR'].values()))[-1]
#
# clov_error_vals = [clov_error_25, clov_error_50, clov_error_75, clov_error_100, clov_error_125]
# dip_error_vals = [dip_error_25, dip_error_50, dip_error_75, dip_error_100, dip_error_125]
#
# standoff_vals = 25*np.arange(1, 6)
#
# fig, axZ1 = plt.subplots()
# axZ1.set_xlabel('Standoff (nm)', fontsize=14)
# axZ1.set_ylabel('Landau Relative Error (%)', fontsize=14, color='b')
# axZ1.scatter(standoff_vals, clov_error_vals, color='b', label='Clover', s=30, marker='^')
# axZ1.locator_params(axis='x', nbins=5)
# axZ1.locator_params(axis='y', nbins=6)
# axZ1.set_ylim([0, 2.4])
# axZ1.tick_params(axis='y', labelcolor='b')
#
# axZ2 = axZ1.twinx()
# axZ2.set_ylabel('Dipole Relative Error (%)', fontsize=14, color='r')
# axZ2.scatter(standoff_vals, dip_error_vals, color='r', label='Dipole', s=30)
# axZ2.locator_params(axis='x', nbins=5)
# axZ2.locator_params(axis='y', nbins=6)
# axZ2.tick_params(axis='y', labelcolor='r')

# fig.tight_layout()
# plt.savefig('standoffFigure.png', dpi=300)
# plt.show()

# plt.scatter(standoff_vals, clov_error_vals, color='red')
# plt.scatter(standoff_vals, dip_error_vals, color='blue')
# plt.locator_params(axis='x', nbins=5)
# plt.tick_params(axis='both', top=True, right=True, direction='in')
# plt.savefig('standoff_comparison.png', dpi=300)
# plt.show()