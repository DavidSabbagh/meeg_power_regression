import os.path as op

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
import mne

import config as cfg
from library.spfiltering import ProjIdentitySpace, ProjCommonSpace
from library.featuring import Riemann, LogDiag  # , RiemannSnp, NaiveVec

meg = 'mag'
n_compo = 65
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 10
n_jobs = 10

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

# defining models
identity = ProjIdentitySpace()

logdiag = LogDiag()

sc = StandardScaler()

dummy = DummyRegressor()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# regression task
print('Pipeline: Dummy')
pipe = make_pipeline(identity, logdiag, sc, dummy)
scores = - cross_val_score(pipe, X=X_leadfield, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_dummy = scores

print('Pipeline: Identity + LogDiag')
scores_id_logdiag = []
pipe = make_pipeline(identity, logdiag, sc, ridge)
for X in [X_leadfield, X_power, X_full]:
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_id_logdiag.append(scores)

print('Pipeline: CommonEucl + RiemannGeo')
scores_unsup_riemanngeo = []
commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=0)
for X in [X_leadfield, X_power, X_full]:
    n_fb = X.shape[1]
    riemanngeo = Riemann(n_fb=n_fb, metric=metric)
    pipe = make_pipeline(commoneucl, riemanngeo, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_unsup_riemanngeo.append(scores)

scores = {'dummy': np.array(scores_dummy),
          'id_logdiag': np.array(scores_id_logdiag),
          'unsup_riemanngeo': np.array(scores_unsup_riemanngeo)}

np.save(op.join(cfg.path_outputs,
                'all_scores_mag_models_mne_connectivity.npy'), scores)
