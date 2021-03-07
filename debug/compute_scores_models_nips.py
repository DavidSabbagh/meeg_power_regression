import os.path as op

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import mne

import config as cfg
from library.spfiltering import (ProjIdentitySpace, ProjCommonSpace,
                                 ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, RiemannSnp

meg = 'mag'
n_compo = 65
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 50
n_jobs = 50

info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg=meg)

fname = op.join(cfg.path_outputs, 'covs_allch_oas.h5')
covs = mne.externals.h5io.read_hdf5(fname)
subjects = [d['subject'] for d in covs if 'subject' in d]
covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d]
X = np.array(covs)
n_sub, n_fb, n_ch, _ = X.shape

part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))
y = part.set_index('Observations').age.loc[subjects]

logdiag = LogDiag()
riemanngeo = Riemann(n_fb=n_fb, metric=metric)
riemannwass = RiemannSnp(n_fb=n_fb, rank=n_compo)

sc = StandardScaler()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
dummy = DummyRegressor()
cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

print('---LogDiag-------------------------------------------')
reg = 0
shrinks = np.linspace(0, 1, 10)

identity = ProjIdentitySpace()
spoc = ProjSPoCSpace(shrink=0.5, scale=scale, n_compo=n_compo, reg=reg)

print('Pipeline: Identity + LogDiag')
pipe = make_pipeline(identity, logdiag, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_sensor_logdiag = scores

print('Pipeline: SPoC + LogDiag')
scores_sup_logdiag = []
for shrink in shrinks:
    spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=reg)
    pipe = make_pipeline(spoc, logdiag, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_sup_logdiag.append(scores)
scores_sup_logdiag = scores_sup_logdiag[np.argmin(np.mean(
                                          scores_sup_logdiag, axis=1))]

print('---RiemannWass-------------------------------------------')
reg = 0
shrinks = np.linspace(0, 1, 10)

identity = ProjIdentitySpace()

print('Pipeline: Identity + RiemannWass')
pipe = make_pipeline(identity, riemannwass, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_sensor_riemannwass = scores

print('---RiemannGeo-------------------------------------------')
shrink = 0.5
regs = np.logspace(-7, -3, 5)

commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=0)

print('Pipeline: CommonEucl + RiemannGeo')
scores_unsup_riemanngeo = []
for reg in regs:
    commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
    pipe = make_pipeline(commoneucl, riemanngeo, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_unsup_riemanngeo.append(scores)
scores_unsup_riemanngeo = scores_unsup_riemanngeo[np.argmin(
                               np.mean(scores_unsup_riemanngeo, axis=1))]

scores = {'sensor_logdiag': np.array(scores_sensor_logdiag),
          'sup_logdiag': np.array(scores_sup_logdiag),
          'sensor_riemannwass': np.array(scores_sensor_riemannwass),
          'unsup_riemanngeo': np.array(scores_unsup_riemanngeo)}

np.save(op.join(cfg.path_outputs, 'all_scores_mag_models_nips.npy'), scores)
