import os.path as op

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
# from sklearn.model_selection import cross_val_score, KFold, GroupShuffleSplit
# from sklearn.model_selection import GridSearchCV
# from sklearn.base import clone
import mne
import pandas as pd

from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path

import config as cfg
from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, NaiveVec

from joblib import Parallel, delayed
##############################################################################
n_compo = 151
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 10
n_jobs = 20

##############################################################################
# Define parameters
fname = data_path() + '/SubjectCMC.ds'
# may need mkdir ~/mne_data
# import locale; locale.setlocale(locale.LC_ALL, "en_US.utf8")

raw = mne.io.read_raw_ctf(fname)
raw.crop(450., 650.).load_data()  # crop for memory purposes
# (0-400s=lft=> crop 50-250, 400-800=rgt => crop 450-650)

# Filter muscular activity to only keep high frequencies
emg = raw.copy().pick_channels(['EMGrgt'])
emg.filter(20., None, fir_design='firwin')

# Filter MEG data to focus on beta band, no ref channels (!)
raw.pick_types(meg=True, ref_meg=False, eeg=False, eog=False)
raw.filter(15., 30., fir_design='firwin')

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=1.5)

# Epoch length is 1.5 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1.500, baseline=None,
                    detrend=1, decim=8, preload=True)
emg_epochs = Epochs(emg, events, tmin=0., tmax=1.500, baseline=None)

# Prepare data
X = np.array([mne.compute_covariance(
              meg_epochs[ii], method='oas')['data'][None]
              for ii in range(len(meg_epochs))])

y = emg_epochs.get_data().var(axis=2)[:, 0]  # target is EMG power

n_sub, n_fb, n_ch, _ = X.shape

##############################################################################

ridge_shrinkage = np.logspace(-3, 5, 100)
spoc_shrinkage = np.linspace(0, 1, 5)
common_shrinkage = np.logspace(-7, -3, 5)

pipelines = {
    'dummy':  make_pipeline(
        ProjIdentitySpace(), LogDiag(), StandardScaler(), DummyRegressor()),
    'naive': make_pipeline(ProjIdentitySpace(), NaiveVec(method='upper'),
                           StandardScaler(),
                           RidgeCV(alphas=ridge_shrinkage)),
    'log-diag': make_pipeline(ProjIdentitySpace(), LogDiag(),
                              StandardScaler(),
                              RidgeCV(alphas=ridge_shrinkage)),
    'spoc': make_pipeline(
            ProjSPoCSpace(n_compo=n_compo,
                          scale=scale, reg=0, shrink=0.5),
            LogDiag(),
            StandardScaler(),
            RidgeCV(alphas=ridge_shrinkage)),
    'riemann':
        make_pipeline(
            ProjCommonSpace(scale=scale, n_compo=n_compo, reg=1.e-05),
            Riemann(n_fb=n_fb, metric=metric),
            StandardScaler(),
            RidgeCV(alphas=ridge_shrinkage))
}

n_components = np.arange(1, 152, 1)
# now let's do group shuffle split
splits = np.array_split(np.arange(len(y)), n_splits)
groups = np.zeros(len(y), dtype=np.int)
for val, inds in enumerate(splits):
    groups[inds] = val


def run_low_rank(n_components, X, y, cv, estimators, scoring, groups):
    out = dict(n_components=n_components)
    for name, est in estimators.items():
        print(name)
        this_est = est
        this_est.steps[0][1].n_compo = n_components
        scores = cross_val_score(
            X=X, y=y, cv=cv, estimator=this_est, n_jobs=1,
            groups=groups,
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
                    groups=groups,
                    cv=GroupShuffleSplit(
                        n_splits=10, train_size=.8, test_size=.2),
                    estimators=low_rank_estimators, scoring='r2')
                    for cc in n_components)
out_frames = list()
for this_dict in out_list:
    this_df = pd.DataFrame({'spoc': this_dict['spoc'],
                           'riemann': this_dict['riemann']})
    this_df['n_components'] = this_dict['n_components']
    this_df['fold_idx'] = np.arange(len(this_df))
    out_frames.append(this_df)
out_df = pd.concat(out_frames)
out_df.to_csv("./outputs/fieldtrip_component_scores.csv")

mean_df = out_df.groupby('n_components').mean().reset_index()
best_components = {
    'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
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

for scoring in ("r2", "neg_mean_absolute_error"):
    # now regular buisiness
    all_scores = dict()
    for key, estimator in pipelines.items():
        cv = GroupShuffleSplit(n_splits=n_splits, train_size=.8, test_size=.2)
        scores = cross_val_score(X=X, y=y, estimator=estimator,
                                 cv=cv, n_jobs=min(n_splits, n_jobs),
                                 groups=groups,
                                 scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        all_scores[key] = scores
    score_name = scoring if scoring == 'r2' else 'mae'
    np.save(op.join(cfg.path_outputs,
                    f'all_scores_models_fieldtrip_spoc_{score_name}.npy'),
            all_scores)


#  # to plot corresponding SPoC patterns
#  spoc = ProjSPoCSpace(n_compo=4, scale=scale, reg=0, shrink=0.5)
#  spoc.fit(X,y)
#  fig = spoc.plot_patterns(meg_epochs.info, fband=0,
#                           name_format='SPoC%01d', scalings=dict(mag=1))
#  fig.savefig('/home/dsabbagh/spoc.png', dpi=300)
#
#  # to check it has similar perf
#  scores=list()
#  for n_compo in range(151):
#      pipelines['spoc'].steps[0][1].n_compo=n_compo
#      estimator=pipelines['spoc']
#      scoring='r2'
#      cv = GroupShuffleSplit(n_splits=n_splits, train_size=.8, test_size=.2)
#      score = cross_val_score(X=X, y=y, estimator=estimator, cv=cv,
#              n_jobs=min(n_splits, n_jobs), groups=groups, scoring=scoring)
#      print(score.mean())
#      scores.append(score.mean())
