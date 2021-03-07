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

n_compo = 21
n_components = np.arange(n_compo)+1
scale = 'auto'
metric = 'riemann'
shrink = .5  # to regularize SPoC
seed = 42
test_size = .1
n_splits = 100
n_jobs = 40

fname = op.join(cfg.derivative_path, 'covs_tuh_oas.h5')
covs = mne.externals.h5io.read_hdf5(fname)
X = np.array([d['covs'] for d in covs if 'subject' in d and d['age'] >= 10])
y = np.array([d['age'] for d in covs if 'subject' in d and d['age'] >= 10])
n_sub, n_fb, n_ch, _ = X.shape

ridge_shrinkage = np.logspace(-3, 5, 100)

pipelines = {
    'dummy': make_pipeline(
                            ProjIdentitySpace(),
                            LogDiag(),
                            StandardScaler(),
                            DummyRegressor()
                            ),
    'naive': make_pipeline(
                            ProjIdentitySpace(),
                            NaiveVec(method='upper'),
                            StandardScaler(),
                            RidgeCV(alphas=ridge_shrinkage)
                            ),
    'log-diag': make_pipeline(
                               ProjIdentitySpace(),
                               LogDiag(),
                               StandardScaler(),
                               RidgeCV(alphas=ridge_shrinkage)
                               ),
    'spoc': make_pipeline(
                           ProjSPoCSpace(n_compo=n_compo, scale=scale,
                                         reg=0, shrink=shrink),
                           LogDiag(),
                           StandardScaler(),
                           RidgeCV(alphas=ridge_shrinkage)
                           ),
    'riemann': make_pipeline(
                              ProjCommonSpace(scale=scale, n_compo=n_compo,
                                              reg=1.e-05),
                              Riemann(n_fb=n_fb, metric=metric),
                              StandardScaler(),
                              RidgeCV(alphas=ridge_shrinkage)
                              )
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
    cv=ShuffleSplit(test_size=.1, n_splits=10, random_state=seed),
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
out_df.to_csv(op.join(cfg.path_outputs, "tuh_component_scores.csv"))

mean_df = out_df.groupby('n_components').mean().reset_index()
best_components = {
    'spoc': mean_df['n_components'][mean_df['spoc'].argmin()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmin()]
}

pipelines[f"spoc_{best_components['spoc']}"] = make_pipeline(
    ProjSPoCSpace(n_compo=best_components['spoc'],
                  scale=scale, reg=0, shrink=shrink),
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

np.save(op.join(cfg.path_outputs,
                f'all_scores_models_tuh_{score_name}_{cv_name}.npy'),
        all_scores)
