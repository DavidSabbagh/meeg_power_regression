import os.path as op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#  from sklearn.dummy import DummyRegressor
#  from sklearn.pipeline import make_pipeline
#  from sklearn.linear_model import RidgeCV
#  from sklearn.preprocessing import StandardScaler
#  from sklearn.model_selection import KFold, cross_val_score
import mne

import config as cfg
from library.spfiltering import ProjCommonSpace
from library.spfiltering import ProjSPoCSpace
#  from library.spfiltering import (ProjIdentitySpace, ProjCommonSpace,
#                                   ProjSPoCSpace)
#  from library.featuring import Riemann, LogDiag

meg = 'mag'
#  shrinks = np.linspace(0, 1, 10)
#  scale_spoc = 1
#  reg_spoc = 1e-2
#  metric = 'wasserstein'
#  seed = 42
#  n_splits = 40
#  n_jobs = 40

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

compos = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 102]
n_fb = 0
scale = 'auto'
reg = 0
shrink = 0.55

med_common = []
med_spoc = []
for n_compo in compos:
    print(n_compo)
    common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
    Xc = common.fit(X, y).transform(X)
    cond_common = []
    for ii in range(n_sub):
        d, V = np.linalg.eigh(Xc[ii, n_fb])
        condition = np.log10(np.abs(d).max()) - np.log10(np.abs(d).min())
        cond_common.append(condition)
    median_common = np.median(cond_common)

    spoc = ProjSPoCSpace(shrink, scale=scale, n_compo=n_compo, reg=reg)
    Xs = spoc.fit(X, y).transform(X)
    cond_spoc = []

    for ii in range(n_sub):
        d, V = np.linalg.eigh(Xs[ii, n_fb])
        condition = np.log10(np.abs(d).max()) - np.log10(np.abs(d).min())
        cond_spoc.append(condition)
    median_spoc = np.median(cond_spoc)
    med_common.append(median_common)
    med_spoc.append(median_spoc)

plt.close('all')
fig = plt.plot(compos, np.vstack([med_common, med_spoc]).T)
plt.xticks(compos)
plt.legend(fig, labels = ['common', 'spoc'])
plt.show()

#  identity = ProjIdentitySpace()
#  logdiag = LogDiag()
#  riemann = Riemann(n_fb=n_fb, metric=metric)
#  sc = StandardScaler()
#  ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
#  dummy = DummyRegressor()
#  cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#
#  print('Pipeline: Common + LogDiag')
#  scores_common_logdiag = []
#  for n_compo in n_compos:
#      common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
#      pipe = make_pipeline(common, logdiag, sc, ridge)
#      scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
#                                 scoring='neg_mean_absolute_error')
#      print("Compo: %d - CV score: %.2f +/- %.2f" % (
#            n_compo, np.mean(scores), np.std(scores)))
#      scores_common_logdiag.append(scores)
#
#  print('Pipeline: Common + Riemann')
#  scores_common_riemann = []
#  for n_compo in n_compos:
#      common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
#      pipe = make_pipeline(common, riemann, sc, ridge)
#      scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
#                                 scoring='neg_mean_absolute_error')
#      print("Compo: %d - CV score: %.2f +/- %.2f" % (
#            n_compo, np.mean(scores), np.std(scores)))
#      scores_common_riemann.append(scores)
#
#  print('Pipeline: SPoC + LogDiag')
#  scores_spoc_logdiag = []
#  for n_compo in n_compos:
#      scoresreg = []
#      for shrink in shrinks:
#          spoc = ProjSPoCSpace(shrink=shrink, scale=scale_spoc,
#                               n_compo=n_compo, reg=reg_spoc)
#          pipe = make_pipeline(spoc, logdiag, sc, ridge)
#          scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
#                                     scoring='neg_mean_absolute_error')
#          print("Compo: %d - Shrink: %.2f - CV score: %.2f +/- %.2f" % (
#                n_compo, shrink, np.mean(scores), np.std(scores)))
#          scoresreg.append(scores)
#      scores_spoc_logdiag.append(scoresreg)
#
#  print('Pipeline: SPoC + Riemann')
#  scores_spoc_riemann = []
#  for n_compo in n_compos:
#      scoresreg = []
#      for shrink in shrinks:
#          spoc = ProjSPoCSpace(shrink=shrink, scale=scale_spoc,
#                               n_compo=n_compo, reg=reg_spoc)
#          pipe = make_pipeline(spoc, riemann, sc, ridge)
#          scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
#                                     scoring='neg_mean_absolute_error')
#          print("Compo: %d - Shrink: %.2f - CV score: %.2f +/- %.2f" % (
#                n_compo, shrink, np.mean(scores), np.std(scores)))
#          scoresreg.append(scores)
#      scores_spoc_riemann.append(scoresreg)
#
#  scores = {'compos': np.array(n_compos), 'shrinks': np.array(shrinks),
#            'dummy': np.array(scores_dummy),
#            'id_logdiag': np.array(scores_id_logdiag),
#            'common_logdiag': np.array(scores_common_logdiag),
#            'common_riemann': np.array(scores_common_riemann),
#            'spoc_logdiag': np.array(scores_spoc_logdiag),
#            'spoc_riemann': np.array(scores_spoc_riemann)}
#
#  np.save(op.join(cfg.path_outputs,
#          'all_scores_mag_scale22_reg7_ridge.npy'), scores)
