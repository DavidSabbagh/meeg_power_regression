import os.path as op
import mne
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
  
import config as cfg
from library.spfiltering import (ProjIdentitySpace, ProjCommonSpace,
                                 ProjSPoCSpace, ProjRandomSpace,
                                 ProjCommonWassSpace, ProjLWSpace)
from library.featuring import Riemann, LogDiag, RiemannSnp

##############################################################################
# Global parameters

n_compo = 13  # see: plot_lemon_cov_spectra.pye
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 10
n_jobs = 1

##############################################################################
# prepare input data

lemon_meta = pd.read_csv(
    "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")
lemon_meta["Age_mean"] = [
    np.array(x).astype(int).mean() for x in lemon_meta.Age.str.split("-")]


with open('./eeg_file_list.txt', 'r') as fid:
    file_list = fid.read().split('\n')[:-1]

subjects = list({ff.split('_')[0] for ff in file_list})

bands = ["low",
         "delta",
         "theta",
         "alpha",
         "beta_low",
         "beta_high",
         "gamma_lo",
         "gamma_mid",
         "gamma_high"]

covs = list()
for subject in subjects:
    sub_covs = list()
    for band in bands:
        cov = mne.read_cov(f"./derivatives/{subject}-{band}-cov.fif")
        sub_covs.append(cov["data"])
    covs.append(sub_covs)

X = np.array(covs)
n_sub, n_fb, n_ch, _ = X.shape

y = lemon_meta.set_index('ID').Age_mean.loc[subjects].values

##############################################################################
# Models

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
random = ProjRandomSpace(n_compo=n_compo)
commonwass = ProjCommonWassSpace(n_compo=n_compo)
commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
spoc = ProjSPoCSpace(shrink=0.5, scale=scale, n_compo=n_compo, reg=reg)

print('Pipeline: Dummy')
pipe = make_pipeline(identity, logdiag, sc, dummy)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_dummy = scores

print('Pipeline: Identity + LogDiag')
pipe = make_pipeline(identity, logdiag, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_sensor_logdiag = scores

print('Pipeline: Random + LogDiag')
pipe = make_pipeline(random, logdiag, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_rand_logdiag = scores

print('Pipeline: CommonEucl + LogDiag')
pipe = make_pipeline(commoneucl, logdiag, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_unsup_logdiag = scores

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
random = ProjRandomSpace(n_compo=n_compo)
commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
spoc = ProjSPoCSpace(shrink=0.5, scale=scale, n_compo=n_compo, reg=reg)

print('Pipeline: Identity + RiemannWass')
pipe = make_pipeline(identity, riemannwass, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_sensor_riemannwass = scores

print('Pipeline: Random + RiemannWass')
pipe = make_pipeline(random, riemannwass, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_rand_riemannwass = scores

print('Pipeline: CommonEucl + RiemannWass')
pipe = make_pipeline(commoneucl, riemannwass, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_unsup_riemannwass = scores

print('Pipeline: SPoC + RiemannWass')
scores_sup_riemannwass = []
for shrink in shrinks:
    spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=reg)
    pipe = make_pipeline(spoc, riemannwass, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_sup_riemannwass.append(scores)
scores_sup_riemannwass = scores_sup_riemannwass[np.argmin(np.mean(
                                                scores_sup_riemannwass, axis=1))]

print('---RiemannGeo-------------------------------------------')
shrink = 0.5
regs = np.logspace(-7, -3, 5)

identity = ProjIdentitySpace()
random = ProjRandomSpace(n_compo=n_compo)
lw = ProjLWSpace(shrink=shrink)
commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=0)
spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=0)

print('Pipeline: Random + RiemannGeo')
pipe = make_pipeline(random, riemanngeo, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_rand_riemanngeo = scores

print('Pipeline: LW + RiemannGeo')
pipe = make_pipeline(lw, riemanngeo, sc, ridge)
scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                           scoring='neg_mean_absolute_error',
                           error_score=np.nan)
print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
scores_sensor_riemanngeo = scores

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

print('Pipeline: SPoC + RiemannGeo')
scores_sup_riemanngeo = []
for reg in regs:
    spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=reg)
    pipe = make_pipeline(spoc, riemanngeo, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_sup_riemanngeo.append(scores)
scores_sup_riemanngeo = scores_sup_riemanngeo[np.argmin(
                               np.mean(scores_sup_riemanngeo, axis=1))]

scores = {'dummy': np.array(scores_dummy),
          'sensor_logdiag': np.array(scores_sensor_logdiag),
          'rand_logdiag': np.array(scores_rand_logdiag),
          'unsup_logdiag': np.array(scores_unsup_logdiag),
          'sup_logdiag': np.array(scores_sup_logdiag),
          'sensor_riemannwass': np.array(scores_sensor_riemannwass),
          'rand_riemannwass': np.array(scores_rand_riemannwass),
          'unsup_riemannwass': np.array(scores_unsup_riemannwass),
          'sup_riemannwass': np.array(scores_sup_riemannwass),
          'sensor_riemanngeo': np.array(scores_sensor_riemanngeo),
          'rand_riemanngeo': np.array(scores_rand_riemanngeo),
          'unsup_riemanngeo': np.array(scores_unsup_riemanngeo),
          'sup_riemanngeo': np.array(scores_sup_riemanngeo)}

np.save(op.join(cfg.path_outputs, 'all_scores_lemon_eeg_models.npy'), scores)
