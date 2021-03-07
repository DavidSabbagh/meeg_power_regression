import os.path as op

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import mne
from pyriemann.tangentspace import TangentSpace

import config_drago as cfg

meg = 'mag'
scale = 1e22
rank = 65
reg = 1e-6
seed = 42
n_jobs = 10
cv = KFold(n_splits=n_jobs, shuffle=True, random_state=seed)


def proj_covs_common(covs, picks, scale=scale, rank=rank, reg=reg):
    covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d]
    covs = scale * np.array(covs)
    n_sub, n_fb, n_ch, n_ch = covs.shape

    # covs2 = covs.reshape(n_sub*n_fb, n_ch, n_ch)
    # covs_avg = np.mean(covs2, axis=0)
    covs_avg = covs.mean(axis=1).mean(axis=0)
    d, V = np.linalg.eigh(covs_avg)
    d = d[::-1]
    V = V[:, ::-1]
    proj_mat = V[:, :rank].T

    covs_proj = np.zeros((n_sub, n_fb, rank, rank))
    for sub in range(n_sub):
        for fb in range(n_fb):
            covs_proj[sub, fb] = proj_mat @ covs[sub, fb] @ proj_mat.T
            covs_proj[sub, fb] += reg * np.eye(rank)
    return covs_proj


def proj_covs_ts(covs):
    n_sub, n_fb, p, _ = covs.shape
    covs_ts = np.zeros((n_sub, n_fb, (p*(p+1))//2))
    for fb in range(n_fb):
        covs_ts[:, fb, :] = TangentSpace(metric="wasserstein").fit(
                covs[:, fb, :, :]).transform(covs[:, fb, :, :])
    return covs_ts


file_covs = op.join(cfg.path_outputs, 'covs_allch_oas.float32.h5')
covs_allch = mne.externals.h5io.read_hdf5(file_covs)  # (sub, fb, ch, ch)

info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg=meg)

covs = proj_covs_common(covs_allch, picks, scale=scale, rank=rank, reg=reg)
X = proj_covs_ts(covs)
X = X.reshape(len(X), -1)

info = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))
subjects = [d['subject'] for d in covs_allch if 'subject' in d]
y = info.set_index('Observations').age.loc[subjects]

ridge = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=np.logspace(-3, 5, 100)))
score = - cross_val_score(ridge, X, y, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=n_jobs,
                          verbose=True)
