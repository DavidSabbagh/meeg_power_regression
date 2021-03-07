import os.path as op

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
import mne

import config as cfg
from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag  # , RiemannSnp, NaiveVec

meg = 'mag'
n_compo = 65
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 100
n_jobs = 40
ridge_shrinkage = np.logspace(-3, 5, 100)

info = np.load(op.join(cfg.path_data, 'info_allch.npy'),
               allow_pickle=True).item()
picks = mne.pick_types(info, meg=meg)

fname = op.join(cfg.path_data, 'covs_allch_oas.h5')
covs = mne.externals.h5io.read_hdf5(fname)
subjects = [d['subject'] for d in covs if 'subject' in d]

# check if mne subjects are all in ggt subjects
subjects_mne = np.load(op.join(cfg.path_outputs,
                       'scores_mag_models_mne_subjects.npy'),
                       allow_pickle=True)
ggt_fname = op.join(cfg.path_data, 'GGT_mne.h5')
ggt = mne.externals.h5io.read_hdf5(ggt_fname)
subjects_ggt = [sub['subject'] for sub in ggt if sub['error'] == 'None']
assert np.all([sub in subjects_ggt for sub in subjects_mne])

# find common subjects between mne and cov
subjects_common = [sub for sub in subjects_mne if sub in subjects]

# forming X
covs_leadfield = [d['ggt'][picks, :][:, picks]
                  for d in ggt
                  if 'subject' in d and d['subject'] in subjects_common]
X_leadfield = np.array(covs_leadfield)[:, None, :, :]

covs_full = [d['covs'][:, picks][:, :, picks]
             for d in covs
             if 'subject' in d and d['subject'] in subjects_common]
X_full = np.array(covs_full)
n_sub, n_fb, n_ch, _ = X_full.shape

X_power = np.empty((n_sub, n_fb, n_ch, n_ch))
for ii, sub in enumerate(X_full):
    traceggt = np.trace(X_leadfield[ii, 0])
    for kk, fb in enumerate(sub):
        power = np.trace(fb)/traceggt
        X_power[ii, kk] = power * X_leadfield[ii, 0]

# forming y
part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))
y = part.set_index('Observations').age.loc[subjects_common]

pipelines = {
    'log-diag': make_pipeline(ProjIdentitySpace(), LogDiag(),
                              StandardScaler(),
                              RidgeCV(alphas=ridge_shrinkage)),
    'spoc': make_pipeline(
        ProjSPoCSpace(n_compo=n_compo,
                      scale=scale, reg=0, shrink=0.5),
        LogDiag(),
        StandardScaler(),
        RidgeCV(alphas=ridge_shrinkage)),
    'riemann': make_pipeline(
        ProjCommonSpace(scale=scale, n_compo=n_compo, reg=1.e-05),
        Riemann(n_fb=n_fb, metric=metric),
        StandardScaler(),
        RidgeCV(alphas=ridge_shrinkage))
}
# add best pipelines
best_df = pd.read_csv("./outputs/camcan_component_scores.csv")
mean_df = best_df.groupby('n_components').mean().reset_index()
best_components = {
    'spoc': mean_df['n_components'][mean_df['spoc'].argmin()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmin()]
}

pipelines[f"spoc_{best_components['spoc']}"] = make_pipeline(
    ProjSPoCSpace(n_compo=best_components['spoc'],
                  scale=scale, reg=0, shrink=0.5),
    LogDiag(),
    StandardScaler(),
    RidgeCV(alphas=ridge_shrinkage))

pipelines[f"riemann_{best_components['riemann']}"] = make_pipeline(
    ProjCommonSpace(scale=scale, n_compo=best_components['riemann'],
                    reg=1.e-05),
    Riemann(n_fb=n_fb, metric=metric),
    StandardScaler(),
    RidgeCV(alphas=ridge_shrinkage))

scoring = 'neg_mean_absolute_error'

out_fname = op.join(cfg.path_outputs,
                    'all_scores_camcan_error_decomposition.npy')

all_scores = dict()
for kind, X in zip(('full', 'power', 'leadfield'),
                   (X_full, X_power, X_leadfield)):
    # handle different dimensions across simus.]
    all_scores[kind] = dict()
    for key, estimator in pipelines.items():
        estimator.steps[1][1].n_fb = X.shape[1]
        cv = ShuffleSplit(test_size=.1, n_splits=n_splits, random_state=seed)
        scores = cross_val_score(X=X, y=y, estimator=estimator,
                                 cv=cv, n_jobs=min(n_splits, n_jobs),
                                 scoring=scoring)

        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        all_scores[kind][key] = scores

np.save(out_fname, all_scores)
