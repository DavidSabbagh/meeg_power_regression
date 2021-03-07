import os.path as op

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.metrics import mutual_info_score
import mne

import config as cfg
import library.preprocessing as pp
from library.spfiltering import ProjSPoCSpace

meg = 'mag'
n_compo = 65
scale = 'auto'
reg = 0
shrink = 0.55
seed = 42
n_splits = 10
n_jobs = 10


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg=meg)

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

bestspocreg = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo,
                            reg=reg)
bestspocreg.fit(X, y)
features = bestspocreg.transform(X)
features = np.array([[np.log10(np.diag(fb)) for fb in sub]
                     for sub in features])
plt.close('all')

# f-regression
plt.figure()
feats = features.reshape(len(features), -1)
F, pv = f_regression(feats, y)
pv = pv.reshape(n_compo, -1)
F = F.reshape(n_compo, -1)
cols = plt.cm.viridis(np.linspace(0, 1, len(pp.fbands)))
# for x, col in zip(-np.log10(pv), cols):
for x, col in zip(F.T, cols):
    plt.plot(np.sort(x)[::-1], color=col)
plt.legend(labels=pp.fbands)
plt.xlabel('Number of components')
plt.ylabel('F-score')

# pearson correlation
features = features.reshape(len(X), 70, len(cfg.fbands))
plt.figure()
corr = [[pearsonr(features[:, comp, fb], y)[0]
         for fb in range(len(cfg.fbands))]
        for comp in range(70)]
cols = plt.cm.viridis(np.linspace(0, 1, len(cfg.fbands)))
for x, col in zip(np.array(corr).T, cols):
    plt.plot(np.sort(x)[::-1], color=col)
plt.legend(labels=cfg.fbands)
plt.xlabel('Number of components')
plt.ylabel('Pearson Correlation')

# some subjects have opposite ordering of power!
# suspicion: sorting of eigenvectors is probably somehow broken.
# plt.figure()
# plt.plot(np.arange(70), corr[:, cfg.fbands.index((8.0, 15.0))])
# plt.show()
# plt.figure()
# plt.plot(features[:, 0, 3])
# plt.ylim(features.min(), featu.max())

# mutual information
plt.figure()
corr = [[calc_MI(features[:, comp, fb], y, 100)
         for fb in range(len(cfg.fbands))]
        for comp in range(70)]
cols = plt.cm.viridis(np.linspace(0, 1, len(cfg.fbands)))
for x, col in zip(np.array(corr).T, cols):
    plt.semilogx(np.sort(x)[::-1], color=col)
plt.legend(labels=cfg.fbands)
plt.xlabel('Number of components')
plt.ylabel('Mutual Information')
