import os.path as op

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import mne

import config as cfg
from library.spfiltering import ProjIdentitySpace, ProjCommonSpace
from library.featuring import LogDiag, Riemann

meg = 'mag'
scale = 1e22
n_compo = 71
reg = 1e-7
metric = 'wasserstein'
seed = 42
n_splits = 10
n_jobs = 10

info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg=meg)

durations = [1, 2, 4, 6, 10]
scores_model = []
scores_base = []
for duration in durations:
    if duration == 10:
        fname = op.join(cfg.path_outputs, 'covs_allch_oas.h5')
    else:
        fname = op.join(cfg.path_outputs, 'covs_allch_oas_%d.h5' % duration)
    covs = mne.externals.h5io.read_hdf5(fname)
    subjects = [d['subject'] for d in covs if 'subject' in d]
    covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d]
    X = np.array(covs)
    n_sub, n_fb, n_ch, _ = X.shape
    print(f'Duration: {duration} - #subjects: {n_sub}')

    part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))
    y = part.set_index('Observations').age.loc[subjects]

    identity = ProjIdentitySpace()
    common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
    riemann = Riemann(n_fb=n_fb, metric=metric)
    logdiag = LogDiag()
    sc = StandardScaler()
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pipe = make_pipeline(common, riemann, sc, ridge)
    score = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                              scoring='neg_mean_absolute_error')
    scores_model.append(score)

    pipe = make_pipeline(identity, logdiag, sc, ridge)
    score = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                              scoring='neg_mean_absolute_error')
    scores_base.append(score)

scores = {'durations': np.array(durations),
          'scores_model': np.array(scores_model),
          'scores_base': np.array(scores_base)}
np.save(op.join(cfg.path_outputs,
        'all_scores_duration_mag_scale22_reg7_ridge.npy'), scores)
