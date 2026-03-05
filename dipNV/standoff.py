from dipNV.backend.packages import *

clov_25 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_25nm.json')
clov_35 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_35nm.json')
clov_45 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_45nm.json')
clov_55 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_55nm.json')
clov_65 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_65nm.json')
clov_75 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_75nm.json')
clov_85 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_85nm.json')
clov_95 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_95nm.json')
clov_105 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_105nm.json')
clov_115 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_115nm.json')
# clov_125 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_clov_rot0_125nm.json')

dip_25 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff25nm.json')
dip_35 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff35nm.json')
dip_45 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff45nm.json')
dip_55 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff55nm.json')
dip_65 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff65nm.json')
dip_75 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff75nm.json')
dip_85 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff85nm.json')
dip_95 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff95nm.json')
dip_105 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff105nm.json')
dip_115 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff115nm.json')
# dip_125 = open(r'C:\Users\zande\PycharmProjects\ANL2025\dipNV\output\QOIs\QOI_februn3_dip_rot0_standoff125nm.json')

clov_25_QOI = json.load(clov_25)
clov_35_QOI = json.load(clov_35)
clov_45_QOI = json.load(clov_45)
clov_55_QOI = json.load(clov_55)
clov_65_QOI = json.load(clov_65)
clov_75_QOI = json.load(clov_75)
clov_85_QOI = json.load(clov_85)
clov_95_QOI = json.load(clov_95)
clov_105_QOI = json.load(clov_105)
clov_115_QOI = json.load(clov_115)
# clov_125_QOI = json.load(clov_125)
dip_25_QOI = json.load(dip_25)
dip_35_QOI = json.load(dip_35)
dip_45_QOI = json.load(dip_45)
dip_55_QOI = json.load(dip_55)
dip_65_QOI = json.load(dip_65)
dip_75_QOI = json.load(dip_75)
dip_85_QOI = json.load(dip_85)
dip_95_QOI = json.load(dip_95)
dip_105_QOI = json.load(dip_105)
dip_115_QOI = json.load(dip_115)
# dip_125_QOI = json.load(dip_125)

clov_25_MSE_err = 1e9*np.asarray(list(clov_25_QOI['MSE_LOSS'].values()))
clov_35_MSE_err = 1e9*np.asarray(list(clov_35_QOI['MSE_LOSS'].values()))
clov_45_MSE_err = 1e9*np.asarray(list(clov_45_QOI['MSE_LOSS'].values()))
clov_55_MSE_err = 1e9*np.asarray(list(clov_55_QOI['MSE_LOSS'].values()))
clov_65_MSE_err = 1e9*np.asarray(list(clov_65_QOI['MSE_LOSS'].values()))
clov_75_MSE_err = 1e9*np.asarray(list(clov_75_QOI['MSE_LOSS'].values()))
clov_85_MSE_err = 1e9*np.asarray(list(clov_85_QOI['MSE_LOSS'].values()))
clov_95_MSE_err = 1e9*np.asarray(list(clov_95_QOI['MSE_LOSS'].values()))
clov_105_MSE_err = 1e9*np.asarray(list(clov_105_QOI['MSE_LOSS'].values()))
clov_115_MSE_err = 1e9*np.asarray(list(clov_115_QOI['MSE_LOSS'].values()))
# clov_125_MSE_err = 1e9*np.asarray(list(clov_125_QOI['MSE_LOSS'].values()))
dip_25_MSE_err = 1e9*np.asarray(list(dip_25_QOI['MSE_LOSS'].values()))
dip_35_MSE_err = 1e9*np.asarray(list(dip_35_QOI['MSE_LOSS'].values()))
dip_45_MSE_err = 1e9*np.asarray(list(dip_45_QOI['MSE_LOSS'].values()))
dip_55_MSE_err = 1e9*np.asarray(list(dip_55_QOI['MSE_LOSS'].values()))
dip_65_MSE_err = 1e9*np.asarray(list(dip_65_QOI['MSE_LOSS'].values()))
dip_75_MSE_err = 1e9*np.asarray(list(dip_75_QOI['MSE_LOSS'].values()))
dip_85_MSE_err = 1e9*np.asarray(list(dip_85_QOI['MSE_LOSS'].values()))
dip_95_MSE_err = 1e9*np.asarray(list(dip_95_QOI['MSE_LOSS'].values()))
dip_105_MSE_err = 1e9*np.asarray(list(dip_105_QOI['MSE_LOSS'].values()))
dip_115_MSE_err = 1e9*np.asarray(list(dip_115_QOI['MSE_LOSS'].values()))
# dip_125_MSE_err = 1e9*np.asarray(list(dip_125_QOI['MSE_LOSS'].values()))


standoff = [25, 35, 45, 55, 65, 75, 85, 95, 105, 115]
clov_MSE = [
    clov_25_MSE_err[-1],
    clov_35_MSE_err[-1],
    clov_45_MSE_err[-1],
    clov_55_MSE_err[-1],
    clov_65_MSE_err[-1],
    clov_75_MSE_err[-1],
    clov_85_MSE_err[-1],
    clov_95_MSE_err[-1],
    clov_105_MSE_err[-1],
    clov_115_MSE_err[-1],
    # clov_125_MSE_err[-1],
]
dip_MSE = [
    dip_25_MSE_err[-1],
    dip_35_MSE_err[-1],
    dip_45_MSE_err[-1],
    dip_55_MSE_err[-1],
    dip_65_MSE_err[-1],
    dip_75_MSE_err[-1],
    dip_85_MSE_err[-1],
    dip_95_MSE_err[-1],
    dip_105_MSE_err[-1],
    dip_115_MSE_err[-1],
    # dip_125_MSE_err[-1],
]
print(dip_MSE)
print(clov_MSE)
fig, ax1 = plt.subplots()

# Left y-axis: clov
ax1.scatter(standoff, clov_MSE, color='red', label='Landau domain')
# ax1.scatter(standoff, dip_MSE, color='blue', label='Dipole domain')
ax1.set_xlabel('Standoff (nm)', fontsize=12)
ax1.set_ylabel('Clover MSE Error (nT$^2$)', fontsize=12)
ax1.tick_params(axis='y')
ax1.locator_params(axis='x', nbins=10)

# Right y-axis: dip
# ax2 = ax1.twinx()
# ax2.scatter(standoff, dip_MSE, color='blue', label='Dipole domain', marker='^')
# ax2.set_ylabel('Dip MSE Error (nT$^2$)', fontsize=12, color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# Optional: combined legend
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
plt.subplots_adjust(top=0.25)
ymin1, ymax1 = ax1.get_ylim()
ax1.set_ylim(ymin1, ymax1 * 1.05)
# ymin2, ymax2 = ax2.get_ylim()
# ax2.set_ylim(ymin2, ymax2 * 1.05)
# ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig('clover_standoff.png', dpi=300)
plt.show()