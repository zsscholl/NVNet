import matplotlib.pyplot as plt
from dipNV.backend.packages import *
from dipNV.backend.utils import *
from dipNV.backend.forward_model import *
from dipNV.backend.deprojection import *
from dipNV.backend.model import *

class TrainDIP(nn.Module):
    def __init__(self, loaded):
        super().__init__()

        # INITIALIZING THE MODEL
        torch.manual_seed(42)
        self.model = NVNet(loaded.CONFIG['ML']['DEPTH'], loaded.CONFIG['ML']['DO_CLAMPED_RELU']).to(device=loaded.device)
        self.fm = forwardModel(loaded)
        self.MSEloss = nn.MSELoss()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=loaded.CONFIG['ML']['INIT_LR'])
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=30, factor=0.5, threshold=1e-9)

        # SETTING UP A DICT TO SAVE QUANTITIES OF INTEREST
        self.QOI = dict()
        self.QOI['MSE_LOSS'] = dict()
        self.QOI['DIV_LOSS'] = dict()
        self.QOI['REL_ERR'] = dict()
        self.QOI['AVG_MAG'] = dict()
        self.QOI['SNR'] = dict()

        # LOADING DATA AND MASKING
        self.stray = None
        if loaded.test_stray is None:
            self.stray = deprojector(loaded).iterative_deprojection(2000, 0.01, 500)
        if loaded.test_stray is not None:
            self.stray = loaded.test_stray
        self.input = torch.load(f'{ROOT}' + f'/dipNV/backend/dip_input_512.pt')
        if loaded.CONFIG['ML']['SAVE_NV'] is True:
            np.save(f'{ROOT}' + f'/dipNV/{loaded.CONFIG['SAVE_NAME']}_nv_array.npy', toNumpy(self.stray))
        self.mask = None
        if loaded.source_mask is not None:
            self.mask = loaded.source_mask*loaded.CONFIG['MAT_PARAMS']['M_SAT']
        graph = PlotML(self.stray)
        # strayMagnitude = np.sqrt(toNumpy(self.stray)[0]**2+toNumpy(self.stray)[1]**2+toNumpy(self.stray)[2]**2)
        # plt.imsave('LANDAU_INSET.png', strayMagnitude, cmap='viridis')

        # CORE TRAINING LOOP
        for epoch in tqdm(range(loaded.CONFIG['ML']['EPOCHS'])):
            self.optim.zero_grad()
            if self.mask is not None:
                output = self.model(self.input)*self.mask
                if loaded.CONFIG['ML']['DO_CLAMPED_RELU'] is True:
                    output = torch.clamp(output, min=-loaded.CONFIG['MAT_PARAMS']['M_SAT'],
                                         max=loaded.CONFIG['MAT_PARAMS']['M_SAT'])
            else:
                output = self.model(self.input)*loaded.CONFIG['MAT_PARAMS']['M_SAT']
            forward_stray = self.fm.propagateMag(output)

            # COMPUTING LOSS AND OTHER QOI'S
            MSEloss = loaded.CONFIG['ML']['MSE']*self.MSEloss(forward_stray, self.stray)
            rel_err = MSEloss.item()/torch.mean(self.stray**2)
            SNR = 10*torch.log10(torch.mean(forward_stray**2)/MSEloss)
            x_mask = self.mask[:, 0, :, :].squeeze() != 0
            y_mask = self.mask[:, 1, :, :].squeeze() != 100
            x_mag_roi = output[0, 0, x_mask]
            y_mag_roi = output[0, 1, y_mask]
            if len(y_mag_roi.shape) <= len(x_mag_roi.shape):
                avg_mag = torch.mean(torch.abs(x_mag_roi))
            else:
                avg_mag = torch.mean(torch.sqrt(x_mag_roi**2 + y_mag_roi**2))
            loss = MSEloss
            self.QOI['MSE_LOSS'].update({epoch: MSEloss.item()})
            self.QOI['REL_ERR'].update({epoch: rel_err.item()})
            self.QOI['AVG_MAG'].update({epoch: avg_mag.item()})
            self.QOI['SNR'].update({epoch: SNR.item()})

            # CLOSING THE LOOP
            MSEloss.backward()
            self.optim.step()
            self.sched.step(loss.item())
            if loaded.CONFIG['ML']['DISPLAY_RATE'] is not None:
                if epoch % loaded.CONFIG['ML']['DISPLAY_RATE'] == 0:
                    print(f'Epoch: {epoch}, MSELoss: {MSEloss.item()}, RelativeError: {rel_err.item()}'
                          +'\n' + f'LR: {self.sched.get_last_lr()}, avg_mag: {avg_mag}, SNR: {SNR}')

                    graph.Render(output, forward_stray)

        torch.save(self.model.state_dict(), f'{ROOT}'+f'/dipNV/output/models/{loaded.CONFIG['SAVE_NAME']}.pth')
        with open(f'{ROOT}' + f'/dipNV/output/QOIs/QOI_{loaded.CONFIG['SAVE_NAME']}.json', 'w') as file:
            json.dump(self.QOI, file)
        with open(f'{ROOT}'+f'/dipNV/output/config_dicts/CONFIG_{loaded.CONFIG['SAVE_NAME']}.json', 'w') as file:
            json.dump(loaded.CONFIG, file)

class EvalDIP():
    def __init__(self, model, loaded):
        torch.manual_seed(42)
        self.model = model.eval()
        self.fm = forwardModel(loaded)
        if loaded.test_stray is not None:
            self.stray = loaded.test_stray
        else:
            self.stray = deprojector(loaded).iterative_deprojection(6000, 0.001, 300)
        self.input = torch.load(f'{ROOT}'+f'/dipNV/backend/dip_input_512.pt')
        self.mask = None
        if loaded.source_mask is not None:
            self.mask = loaded.source_mask
            with torch.no_grad():
                output = torch.clamp(self.model(self.input)*self.mask*loaded.CONFIG['MAT_PARAMS']['M_SAT'], max=loaded.CONFIG['MAT_PARAMS']['M_SAT'])
        else:
            output = torch.clamp(self.model(self.input), max=loaded.CONFIG['MAT_PARAMS']['M_SAT'])
        forward_stray = self.fm.propagateMag(output)
        bx = toNumpy(self.stray)[0]
        by = toNumpy(self.stray)[1]
        bz = toNumpy(self.stray)[2]
        mx = toNumpy(output)[0]
        my = toNumpy(output)[1]
        mz = toNumpy(output)[2]
        abx = toNumpy(forward_stray)[0]
        aby = toNumpy(forward_stray)[1]
        abz = toNumpy(forward_stray)[2]
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/Bx_{loaded.CONFIG['SAVE_NAME']}.npy', bx)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/By_{loaded.CONFIG['SAVE_NAME']}.npy', by)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/Bz_{loaded.CONFIG['SAVE_NAME']}.npy', bz)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/AMx_{loaded.CONFIG['SAVE_NAME']}.npy', abx)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/AMy_{loaded.CONFIG['SAVE_NAME']}.npy', aby)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/AMz_{loaded.CONFIG['SAVE_NAME']}.npy', abz)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/Mx_{loaded.CONFIG['SAVE_NAME']}.npy', mx)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/My_{loaded.CONFIG['SAVE_NAME']}.npy', my)
        np.save(f'{ROOT}'+f'/dipNV/output/arrays/Mz_{loaded.CONFIG['SAVE_NAME']}.npy', mz)

        plt.imshow(bx, cmap='bwr')
        plt.colorbar()
        plt.clim(vmin=-np.abs(bx).max(), vmax=np.abs(bx).max())
        plt.title(r'$B_x$')
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/Bx_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(by, cmap='bwr')
        plt.colorbar()
        plt.clim(vmin=-np.abs(by).max(), vmax=np.abs(by).max())
        plt.title(r'$B_y$')
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/By_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(bz, cmap='bwr')
        plt.colorbar()
        plt.clim(vmin=-np.abs(bz).max(), vmax=np.abs(bz).max())
        plt.title(r'$B_z$')
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/Bz_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(mx, cmap='bwr', origin='lower')
        plt.tick_params(axis='both', which='major', labelsize=14)
        (cbar := plt.colorbar(label=r'$A/m$')).ax.tick_params(labelsize=14) or cbar.set_label(r'$A/m$', size=18)
        plt.clim(vmin=-np.abs(mx).max(), vmax=np.abs(mx).max())
        plt.title(r'$M_x$', fontsize=20)
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/Mx_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(my, cmap='bwr', origin='lower')
        plt.tick_params(axis='both', which='major', labelsize=14)
        (cbar := plt.colorbar(label=r'$A/m$')).ax.tick_params(labelsize=14) or cbar.set_label(r'$A/m$', size=18)
        plt.clim(vmin=-np.abs(my).max(), vmax=np.abs(my).max())
        plt.title(r'$M_y$', fontsize=20)
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/My_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(mz, cmap='bwr', origin='lower')
        plt.tick_params(axis='both', which='major', labelsize=14)
        (cbar := plt.colorbar(label=r'$A/m$')).ax.tick_params(labelsize=14) or cbar.set_label(r'$A/m$', size=18)
        plt.clim(vmin=-np.abs(mz).max(), vmax=np.abs(mz).max())
        plt.title(r'$M_z$', fontsize=20)
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/Mz_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(abx, cmap='bwr')
        plt.colorbar()
        plt.clim(vmin=-np.abs(abx).max(), vmax=np.abs(abx).max())
        plt.title(r'$\mathcal{A}M_x$')
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/AMx_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(aby, cmap='bwr')
        plt.colorbar()
        plt.clim(vmin=-np.abs(aby).max(), vmax=np.abs(aby).max())
        plt.title(r'$\mathcal{A}M_y$')
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/AMy_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)
        plt.clf()
        plt.imshow(abz, cmap='bwr')
        plt.colorbar()
        plt.clim(vmin=-np.abs(abz).max(), vmax=np.abs(abz).max())
        plt.title(r'$\mathcal{A}M_y$')
        plt.savefig(f'{ROOT}'+f'/dipNV/output/img/AMz_{loaded.CONFIG['SAVE_NAME']}.png',
                    dpi=300)




