import glob
import os.path as op

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedKFold, cross_val_score,
                                     ShuffleSplit)
import mne

from library.spfiltering import (ProjIdentitySpace, ProjCommonSpace,
                                 ProjSPoCSpace, ProjRandomSpace,
                                 ProjCommonWassSpace, ProjLWSpace)
from library.featuring import Riemann, LogDiag, RiemannSnp, NaiveVec


class cfg:
    path_dir = '/storage/local/camcan'
    path_data = op.join(path_dir, 'data')
    path_outputs = op.join('./outputs')
    path_maxfilter_info = op.join(path_dir, 'maxfilter')
    files_raw = sorted(glob.glob(op.join(path_data,
                                         'CC??????/rest/rest_raw.fif')))
    camcan_path = '/storage/store/data/camcan'
    camcan_meg_path = op.join(
        camcan_path, 'camcan47/cc700/meg/pipeline/release004/')
    camcan_meg_raw_path = op.join(camcan_meg_path,
                                  'data/aamod_meg_get_fif_00001')
    mne_camcan_freesurfer_path = (
        '/storage/store/data/camcan-mne/freesurfer')
    derivative_path = ('/storage/inria/agramfor/camcan_derivatives')

meg = 'mag'
n_compo = 65
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 100
n_jobs = 20

info = np.load(op.join(cfg.path_data, 'info_allch.npy'),
               allow_pickle=True).item()
picks = mne.pick_types(info, meg=meg)

fname = op.join(cfg.path_data, 'covs_allch_oas.h5')
covs = mne.externals.h5io.read_hdf5(fname)
subjects = [d['subject'] for d in covs if 'subject' in d]
subjects_mne = np.load(op.join(cfg.derivative_path,
                               'scores_mag_models_mne_intervals_subjects.npy'),
                       allow_pickle=True)
subjects_common = [sub for sub in subjects_mne if sub in subjects]
covs = [d['covs'][:, picks][:, :, picks] for d in covs if 'subject' in d
        and d['subject'] in subjects_common]
X = np.array(covs)
n_sub, n_fb, n_ch, _ = X.shape

part = pd.read_csv(op.join(cfg.path_data, 'participants.csv'))
y = part.set_index('Observations').age.loc[subjects_common]

cvs = [ShuffleSplit(test_size=.1, n_splits=100, random_state=seed),
       RepeatedKFold(n_splits=10,  n_repeats=10, random_state=seed)]


def run_models(cv):
    logdiag = LogDiag()
    naivevec = NaiveVec(method='upper')
    riemanngeo = Riemann(n_fb=n_fb, metric=metric)
    riemannwass = RiemannSnp(n_fb=n_fb, rank=n_compo)

    sc = StandardScaler()
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    dummy = DummyRegressor()
    print('---LogDiag-------------------------------------------')
    reg = 0
    shrinks = np.linspace(0, 1, 10)

    identity = ProjIdentitySpace()
    euclidean_vec = NaiveVec(method='upper')
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

    print('Pipeline: Identity + NaiveVec')
    pipe = make_pipeline(identity, naivevec, sc, ridge)
    scores = cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                            scoring='neg_mean_absolute_error',
                            error_score=np.nan)
    print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
    scores_sensor_naivevec = scores

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

    scores_sup_logdiag = scores_sup_logdiag[np.argmin(np.mean(scores_sup_logdiag, axis=1))]

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
              'sensor_euclidean': np.array(scores_sensor_naivevec),
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
    return scores

for kind, cv in zip(('shuffle-split', 'rep-kfold'), cvs):
    scores = run_models(cv=cv)
    np.save(op.join(cfg.path_outputs,
                    f'all_scores_mag_models_mnecommonsubjects_interval_{kind}.npy'),
            scores)
