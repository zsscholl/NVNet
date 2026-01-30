from dipNV.backend.packages import *

f0 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot0_again.json')
f15 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot15_again.json')
f30 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot30_again.json')
f45 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot45_again.json')
f60 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot60_again.json')
f75 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot75_again.json')
f90 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_102325_clov_rot90_again.json')

QOI_0 = json.load(f0)
QOI_15 = json.load(f15)
QOI_30 = json.load(f30)
QOI_45 = json.load(f45)
QOI_60 = json.load(f60)
QOI_75 = json.load(f75)
QOI_90 = json.load(f90)

epochs = np.asarray(list(QOI_0['L2_LOSS'].keys()), dtype=float)
l2_err_0 = 1e9*np.asarray(list(QOI_0['L2_LOSS'].values()))
l2_err_15 = 1e9*np.asarray(list(QOI_15['L2_LOSS'].values()))
l2_err_30 = 1e9*np.asarray(list(QOI_30['L2_LOSS'].values()))
l2_err_45 = 1e9*np.asarray(list(QOI_45['L2_LOSS'].values()))
l2_err_60 = 1e9*np.asarray(list(QOI_60['L2_LOSS'].values()))
l2_err_75 = 1e9*np.asarray(list(QOI_75['L2_LOSS'].values()))
l2_err_90 = 1e9*np.asarray(list(QOI_90['L2_LOSS'].values()))
SNR_0 = np.asarray(list(QOI_0['SNR'].values()))
SNR_15 = np.asarray(list(QOI_15['SNR'].values()))
SNR_30 = np.asarray(list(QOI_30['SNR'].values()))
SNR_45 = np.asarray(list(QOI_45['SNR'].values()))
SNR_60 = np.asarray(list(QOI_60['SNR'].values()))
SNR_75 = np.asarray(list(QOI_75['SNR'].values()))
SNR_90 = np.asarray(list(QOI_90['SNR'].values()))

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.15)  # Adjust space between axes
ax0.scatter(epochs, l2_err_0, s=1, color='red', label=r'$0\degree$')
ax0.scatter(epochs, l2_err_15, s=1, color='green', label=r'$15\degree$')
ax0.scatter(epochs, l2_err_30, s=1, color='blue', label=r'$30\degree$')
# ax0.scatter(epochs, l2_err_45, s=1, color='orange', label=r'$45\degree$')
# ax0.scatter(epochs, l2_err_60, s=1, color='yellow', label=r'$60\degree$')
ax0.scatter(epochs, l2_err_75, s=1, color='orange', label=r'$75\degree$')
ax0.scatter(epochs, l2_err_90, s=1, color='olive', label=r'$90\degree$')

ax1.scatter(epochs, l2_err_0, s=1, color='red', label=r'$0\degree$')
ax1.scatter(epochs, l2_err_15, s=1, color='green', label=r'$15\degree$')
ax1.scatter(epochs, l2_err_30, s=1, color='blue', label=r'$30\degree$')
# ax1.scatter(epochs, l2_err_45, s=1, color='orange', label=r'$45\degree$')
# ax1.scatter(epochs, l2_err_60, s=1, color='yellow', label=r'$60\degree$')
ax1.scatter(epochs, l2_err_75, s=1, color='orange', label=r'$75\degree$')
ax1.scatter(epochs, l2_err_90, s=1, color='olive', label=r'$90\degree$')

ax0.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))

ax0.set_ylim(500,1000)
ax1.set_ylim(0, 500)
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
fig.supylabel(r'$L_2$ Loss (nT$^2$)', fontsize=14)
fig.supxlabel('Epoch', fontsize=14)
# plt.savefig('clover_GoodErrorCurve.png', bbox_inches='tight', dpi=300, facecolor='white')
# plt.show()

snr_array = [SNR_0[-1], SNR_15[-1], SNR_30[-1], SNR_45[-1], SNR_60[-1], SNR_75[-1], SNR_90[-1]]
l2_array = [l2_err_0[-1], l2_err_15[-1], l2_err_30[-1], l2_err_45[-1], l2_err_60[-1], l2_err_75[-1], l2_err_90[-1]]
ax_snr_l2 = plt.subplot()
ax_snr_l2.scatter(l2_array, snr_array, color='red', label='SNR')

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
plt.show()

# plt.scatter(standoff_vals, clov_error_vals, color='red')
# plt.scatter(standoff_vals, dip_error_vals, color='blue')
# plt.locator_params(axis='x', nbins=5)
# plt.tick_params(axis='both', top=True, right=True, direction='in')
# plt.savefig('standoff_comparison.png', dpi=300)
# plt.show()