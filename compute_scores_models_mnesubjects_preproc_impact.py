import os.path as op

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold
import mne

import config as cfg
import glob
from library.spfiltering import ProjIdentitySpace, ProjCommonSpace, ProjLWSpace
from library.featuring import Riemann, LogDiag, RiemannSnp

meg = 'mag'
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 10
n_jobs = 10
n_fb = 9
n_compo = 65

cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
# cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_splits, random_state=seed)
sc = StandardScaler()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
dummy = DummyRegressor()
logdiag = LogDiag()
riemanngeo = Riemann(n_fb=n_fb, metric=metric)
riemannwass = RiemannSnp(n_fb=n_fb, rank=n_compo)
identity = ProjIdentitySpace()
lw = ProjLWSpace(shrink=0.5)

info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg=meg)

part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))

result = dict()
# fname_covs = op.join(cfg.path_data, 'covs_allch_oas_nopreproc.h5')

files_covs = sorted(glob.glob(op.join(cfg.path_data, 'covs_preproc_impact_*')))
for fname in files_covs:
    preproc = fname.split('impact_')[1].split('.h5')[0]
    print(preproc)
    covs = mne.externals.h5io.read_hdf5(fname)
    subjects = [d['subject'] for d in covs if 'subject' in d]
    subjects_mne = np.load(op.join(cfg.path_outputs,
                           'scores_mag_models_mne_subjects.npy'))
    subjects_common = [sub for sub in subjects_mne if sub in subjects]
    covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d
            and d['subject'] in subjects_common]
    X = np.array(covs)
    n_sub, n_fb, n_ch, _ = X.shape
    y = part.set_index('Observations').age.loc[subjects_common]

    print('\tPipeline: Identity + LogDiag')
    pipe = make_pipeline(identity, logdiag, sc, ridge)
    scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                               scoring='neg_mean_absolute_error',
                               error_score=np.nan)
    print("\tCV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_sensor_logdiag = scores

    print('\tPipeline: CommonEucl + RiemannGeo')
    regs = np.logspace(-7, -3, 5)
    scores_unsup_riemanngeo = []
    for reg in regs:
        commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
        pipe = make_pipeline(commoneucl, riemanngeo, sc, ridge)

        scores = - cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                                   scoring='neg_mean_absolute_error',
                                   error_score=np.nan)
        print("\tCV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
        scores_unsup_riemanngeo.append(scores)
    scores_unsup_riemanngeo = scores_unsup_riemanngeo[np.argmin(
                                   np.mean(scores_unsup_riemanngeo, axis=1))]

    result[preproc] = {'sensor_logdiag': np.array(scores_sensor_logdiag),
                       'unsup_riemanngeo': np.array(scores_unsup_riemanngeo)}

np.save(op.join(cfg.path_outputs,
                'all_scores_mag_models_mnecommonsubjects_preproc_impact.npy'),
        result)
