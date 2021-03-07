import os.path as op

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit
import mne

import config as cfg
import glob
from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, RiemannSnp

meg = 'mag'
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 100
n_jobs = 40
n_fb = 9
n_compo = 65
ridge_shrinkage = np.logspace(-3, 5, 100)

info = np.load(op.join(cfg.path_data, 'info_allch.npy'),
               allow_pickle=True).item()
picks = mne.pick_types(info, meg=meg)

part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))


def read_inputs(fname):
    covs = mne.externals.h5io.read_hdf5(fname)
    subjects = [d['subject'] for d in covs if 'subject' in d]
    subjects_mne = np.load(op.join(cfg.path_outputs,
                                   'scores_mag_models_mne_subjects.npy'),
                           allow_pickle=True)
    subjects_common = [sub for sub in subjects_mne if sub in subjects]
    covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d
            and d['subject'] in subjects_common]
    X = np.array(covs)
    y = part.set_index('Observations').age.loc[subjects_common]
    return X, y


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
files_covs = sorted(glob.glob(op.join(cfg.path_data, 'covs_preproc_impact_*')))

results = dict()
results_fname = op.join(cfg.path_outputs, 'camacan_preproc_impact.npy')

for fname in files_covs:
    preproc = fname.split('impact_')[1].split('.h5')[0]
    print(preproc)
    X, y = read_inputs(fname)

    results[preproc] = dict()
    for key, estimator in pipelines.items():
        cv = ShuffleSplit(test_size=.1, n_splits=n_splits, random_state=seed)
        scores = cross_val_score(X=X, y=y, estimator=estimator,
                                 cv=cv, n_jobs=min(n_splits, n_jobs),
                                 scoring=scoring)

        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        results[preproc][key] = scores

np.save(results_fname, results)
