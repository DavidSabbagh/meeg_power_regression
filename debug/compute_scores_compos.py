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
                                 ProjSPoCSpace, ProjRandomSpace,
                                 ProjCommonWassSpace, ProjLWSpace)
from library.featuring import Riemann, LogDiag, RiemannSnp

meg = 'mag'
n_compos = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 102]
scale = 'auto'
reg = 0
shrinks = np.linspace(0, 1, 10)
metric = 'riemann'
seed = 42
n_splits = 10
n_jobs = 10

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

identity = ProjIdentitySpace()
logdiag = LogDiag()
riemann = Riemann(n_fb=n_fb, metric=metric)
sc = StandardScaler()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
dummy = DummyRegressor()
cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

print('Pipeline: Dummy')
pipe = make_pipeline(identity, logdiag, sc, dummy)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error')
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_dummy = scores

print('Pipeline: Identity + LogDiag')
pipe = make_pipeline(identity, logdiag, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error')
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_id_logdiag = scores

print('Pipeline: LW + Riemann')
scores_lw_riemann = []
for alpha in np.logspace(-2, 0, 5):
    lw = ProjLWSpace(alpha=alpha)
    pipe = make_pipeline(lw, riemann, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("Alpha: %e - CV score: %.2f +/- %.2f" % (
          alpha, np.mean(scores), np.std(scores)))
    scores_lw_riemann.append(scores)

print('Pipeline: Common + LogDiag')
scores_common_logdiag = []
for n_compo in n_compos:
    common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
    pipe = make_pipeline(common, logdiag, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("Compo: %d - CV score: %.2f +/- %.2f" % (
          n_compo, np.mean(scores), np.std(scores)))
    scores_common_logdiag.append(scores)

print('Pipeline: Random + Riemann')
scores_random_riemann = []
for n_compo in n_compos:
    random = ProjRandomSpace(n_compo=n_compo)
    pipe = make_pipeline(random, riemann, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("Compo: %d - CV score: %.2f +/- %.2f" % (
          n_compo, np.mean(scores), np.std(scores)))
    scores_random_riemann.append(scores)

print('Pipeline: CommonWass + Riemann')
scores_commonwass_riemann = []
for n_compo in n_compos:
    commonwass = ProjCommonWassSpace(n_compo=n_compo)
    pipe = make_pipeline(commonwass, riemann, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("Compo: %d - CV score: %.2f +/- %.2f" % (
          n_compo, np.mean(scores), np.std(scores)))
    scores_commonwass_riemann.append(scores)

print('Pipeline: Common + Riemann')
scores_common_riemann = []
for n_compo in n_compos:
    common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
    pipe = make_pipeline(common, riemann, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("Compo: %d - CV score: %.2f +/- %.2f" % (
          n_compo, np.mean(scores), np.std(scores)))
    scores_common_riemann.append(scores)

print('Pipeline: SPoC + LogDiag')
scores_spoc_logdiag = []
for n_compo in n_compos:
    scoresreg = []
    for shrink in shrinks:
        spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo,
                             reg=reg)
        pipe = make_pipeline(spoc, logdiag, sc, ridge)
        scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                                   scoring='neg_mean_absolute_error',
                                   error_score=np.nan)
        print("Compo: %d - Shrink: %.2f - CV score: %.2f +/- %.2f" % (
              n_compo, shrink, np.mean(scores), np.std(scores)))
        scoresreg.append(scores)
    scores_spoc_logdiag.append(scoresreg)

print('Pipeline: SPoC + Riemann')
scores_spoc_riemann = []
for n_compo in n_compos:
    scoresreg = []
    for shrink in shrinks:
        spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo,
                             reg=reg)
        pipe = make_pipeline(spoc, riemann, sc, ridge)
        scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                                   scoring='neg_mean_absolute_error',
                                   error_score=np.nan)
        print("Compo: %d - Shrink: %.2f - CV score: %.2f +/- %.2f" % (
              n_compo, shrink, np.mean(scores), np.std(scores)))
        scoresreg.append(scores)
    scores_spoc_riemann.append(scoresreg)

print('Pipeline: RiemannSnp')
scores_riemannsnp = []
for rank in n_compos:
    riemannsnp = RiemannSnp(n_fb=n_fb, rank=rank)
    pipe = make_pipeline(riemannsnp, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("Compo: %d - CV score: %.2f +/- %.2f" % (
          rank, np.mean(scores), np.std(scores)))
    scores_riemannsnp.append(scores)

scores = {'compos': np.array(n_compos), 'shrinks': np.array(shrinks),
          'dummy': np.array(scores_dummy),
          'id_logdiag': np.array(scores_id_logdiag),
          'common_logdiag': np.array(scores_common_logdiag),
          'common_riemann': np.array(scores_common_riemann),
          'spoc_logdiag': np.array(scores_spoc_logdiag),
          'spoc_riemann': np.array(scores_spoc_riemann),
          'id_riemannsnp': np.array(scores_riemannsnp)}

np.save(op.join(cfg.path_outputs,
        'all_scores_mag_scaleAuto_reg0_ridge.npy'), scores)
#
#  n_compo = 50
#  scale = 'auto'
#  shrink = 0.22
#  reg = 1e-4
#  for reg in np.logspace(-7, -3, 5):
#      #  spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo,
#      #                       reg=reg)
#      common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
#      pipe = make_pipeline(common, riemann, sc, ridge)
#      scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
#                                 scoring='neg_mean_absolute_error',
#                                 error_score=np.nan)
#      print("Reg: %e - Shrink: %.2f - CV score: %.2f +/- %.2f" % (
#            reg, shrink, np.mean(scores), np.std(scores)))
