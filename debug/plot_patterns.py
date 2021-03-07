import os.path as op

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import mne
import matplotlib
import matplotlib.pyplot as plt

import config as cfg
from library.spfiltering import ProjSPoCSpace
from library.featuring import LogDiag
import library.preprocessing as pp

font = {'family': 'normal',
        'size': 16}
matplotlib.rc('font', **font)

meg = 'mag'
n_compo = 65
scale = 'auto'
reg = 0
shrink = 0.55
seed = 42
n_splits = 10
n_jobs = 10

info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg=meg)
info = mne.io.pick.pick_info(info, sel=picks)

fname = op.join(cfg.path_outputs, 'covs_allch_oas.h5')
covs = mne.externals.h5io.read_hdf5(fname)
subjects = [d['subject'] for d in covs if 'subject' in d]
subjects_mne = np.load(op.join(cfg.path_outputs,
                       'scores_mag_models_mne_subjects.npy'))
subjects_common = [sub for sub in subjects_mne if sub in subjects]
covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d
        and d['subject'] in subjects_common]
X = np.array(covs)
n_sub, n_fb, n_ch, _ = X.shape

part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))
y = part.set_index('Observations').age.loc[subjects_common]

logdiag = LogDiag()

sc = StandardScaler()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=reg)
spoc.fit(X, y)

plt.close('all')
fbands = pp.fbands
fbands_names = [r'$1/f$', r'$\delta$', r'$\theta$', r'$\alpha$',
                r'$\beta_{low}$', r'$\beta_{high}$',
                r'$\gamma_{low}$', r'$\gamma_{mid}$', r'$\gamma_{high}$']
fig, ax = plt.subplots(1, 9, figsize=(12, 3))
for ii in range(len(fbands)):
    spoc.plot_patterns(info=info, components=0, fband=ii,
                       name_format='', axes=ax[ii], colorbar=False)
    ax[ii].set_title(fbands_names[ii])
plt.tight_layout()
fig.savefig(op.join(cfg.path_outputs, './figures/plot_firstpattern_spoc.png'), dpi=300)
