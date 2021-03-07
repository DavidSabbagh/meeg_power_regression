
import os.path as op
import copy

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit
import mne
import pandas as pd

import config as cfg
from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, NaiveVec

from joblib import Parallel, delayed

import h5py
from tqdm import tqdm

fbands = ('alpha',
          'beta_high',
          'beta_low',
          'delta',
          'gamma_high',
          'gamma_lo',
          'gamma_mid',
          'low',
          'theta')
n_fb = len(fbands)
n_labels = 448
c_index = np.eye(n_labels, dtype=np.bool)
c_index = np.invert(c_index[np.triu_indices(n_labels)])

scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 100
n_jobs = 10
test_size = .1
n_compo = 65
n_components = np.arange(1, 103, 1)
ridge_shrinkage = np.logspace(-3, 5, 100)


def make_mat(cross, diag):
    n_rows = len(diag)
    C = np.zeros((n_rows, n_rows), dtype=np.float64)
    C[np.triu_indices(n=n_rows, k=1)] = cross
    C += C.T + np.diag(diag)
    return C


sensor_covs = mne.externals.h5io.read_hdf5(op.join(cfg.derivative_path,
                                                   'covs_allch_oas.h5'))
sensor_subjects = ['key_'+d['subject'] for d in sensor_covs if 'subject' in d]

source_covs = h5py.File(op.join(cfg.derivative_path,
                                'all_mne_source_power.h5'))
source_subjects = list(source_covs['h5io'].keys())

#  these source covs have been computed directly from raw signal and stored as
#  upper triangle by the script
#  github.com/OlehKSS/camcan_analysis/blob/develop/scripts/denis/compute_cov_inverse_mne.py
#  │ it does raw->preproch->epoching->filtering->epochs-to-source-vial-MNE->cov
#  different approach than in NIPS/NIMG_before_rebuttal articles to compute MNE
#  logdiag using GiSigmaGi.T but this file was too big to fit into memory
#  so Denis cut it in smaller pieces in
#  https://github.com/OlehKSS/camcan_analysis/blob/develop/scripts/denis/rewrite_outputs.py
#  => only take source_power_cross_{band}.h5

common_subjects = [sub for sub in source_subjects if sub in sensor_subjects]
n_sub = len(common_subjects)

X = np.zeros((n_sub, n_fb, n_labels, n_labels))
for ii, sub in enumerate(tqdm(common_subjects)):
    for kk, band in enumerate(tqdm(fbands)):
        diag = source_covs['h5io'][sub]['key_'+band]['key_power'][:, 0]
        cross = source_covs['h5io'][sub]['key_'+band]['key_cov'][c_index]
        cov = make_mat(cross, diag)
        X[ii, kk] = cov

part = pd.read_csv(op.join(cfg.derivative_path, 'participants.csv'))
short_common_subjects = [sub[4:] for sub in common_subjects]
y = part.set_index('Observations').age.loc[short_common_subjects]

pipelines = {
    'dummy':  make_pipeline(
                            ProjIdentitySpace(),
                            LogDiag(),
                            StandardScaler(),
                            DummyRegressor()),
    'naive': make_pipeline(
                           ProjIdentitySpace(),
                           NaiveVec(method='upper'),
                           StandardScaler(),
                           RidgeCV(alphas=ridge_shrinkage)),
    'log-diag': make_pipeline(
                              ProjIdentitySpace(),
                              LogDiag(),
                              StandardScaler(),
                              RidgeCV(alphas=ridge_shrinkage)),
    'spoc': make_pipeline(
                          ProjSPoCSpace(n_compo=n_compo, scale=scale,
                                        reg=0, shrink=0.5),
                          LogDiag(),
                          StandardScaler(),
                          RidgeCV(alphas=ridge_shrinkage)),
    'riemann': make_pipeline(
                             ProjCommonSpace(scale=scale, n_compo=n_compo,
                                             reg=1.e-05),
                             Riemann(n_fb=n_fb, metric=metric),
                             StandardScaler(),
                             RidgeCV(alphas=ridge_shrinkage))
}


def run_low_rank(n_components, X, y, cv, estimators, scoring):
    out = dict(n_components=n_components)
    for name, est in estimators.items():
        print(name, n_components)
        this_est = est
        this_est.steps[0][1].n_compo = n_components
        scores = cross_val_score(
            X=X, y=y, cv=copy.deepcopy(cv), estimator=this_est,
            n_jobs=1,
            scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        print(np.mean(scores), f"+/-{np.std(scores)}")
        out[name] = scores
    return out


low_rank_estimators = {k: v for k, v in pipelines.items()
                       if k in ('spoc', 'riemann')}

out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
    n_components=cc, X=X, y=y,
    cv=ShuffleSplit(test_size=test_size, n_splits=10, random_state=seed),
    estimators=low_rank_estimators, scoring='neg_mean_absolute_error')
    for cc in n_components)

out_frames = list()
for this_dict in out_list:
    this_df = pd.DataFrame({'spoc': this_dict['spoc'],
                            'riemann': this_dict['riemann']})
    this_df['n_components'] = this_dict['n_components']
    this_df['fold_idx'] = np.arange(len(this_df))
    out_frames.append(this_df)

out_df = pd.concat(out_frames)
out_df.to_csv("./outputs/camcan_source_component_scores.csv")

mean_df = out_df.groupby('n_components').mean().reset_index()
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

# now regular buisiness
all_scores = dict()
score_name, scoring = "mae", "neg_mean_absolute_error"
cv_name = 'shuffle-split'
score_name = 'mae'
for key, estimator in pipelines.items():
    cv = ShuffleSplit(test_size=test_size, n_splits=n_splits,
                      random_state=seed)
    scores = cross_val_score(X=X, y=y, estimator=estimator,
                             cv=cv, n_jobs=min(n_splits, n_jobs),
                             scoring=scoring)
    if scoring == 'neg_mean_absolute_error':
        scores = -scores
    all_scores[key] = scores

np.save(
    op.join(cfg.path_outputs,
            f'all_scores_models_camcan_source_{score_name}_{cv_name}.npy'),
    all_scores)
