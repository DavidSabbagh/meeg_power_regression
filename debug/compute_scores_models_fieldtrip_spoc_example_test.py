import os.path as op

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GroupShuffleSplit
import mne

from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path

import config as cfg
from library.spfiltering import (ProjIdentitySpace, ProjCommonSpace,
                                 ProjSPoCSpace, ProjLWSpace)
from library.featuring import Riemann, LogDiag, RiemannSnp, NaiveVec

##############################################################################
n_compo = 151
scale = 'auto'
metric = 'riemann'
shrink = 0
shrinks = np.linspace(0, 1, 10)
reg = 0
regs = np.logspace(-7, -3, 5)
seed = 42
n_splits = 10
n_jobs = 20

# Define parameters
fname = data_path() + '/SubjectCMC.ds'

raw = mne.io.read_raw_ctf(fname)
raw.crop(50, 250).load_data()  # crop for memory purposes
# raw.crop(350, 700).load_data()  # crop for memory purposes

# Filter muscular activity to only keep high frequencies
emg = raw.copy().pick_channels(['EMGlft'])
emg.filter(20., None, fir_design='firwin')

# Filter MEG data to focus on beta band
raw.pick_types(meg=True, ref_meg=False, eeg=False, eog=False)
raw.filter(15., 30., fir_design='firwin')

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=.250)

# Epoch length is 1.5 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1.500, baseline=None,
                    detrend=1, decim=1, preload=True)
emg_epochs = Epochs(emg, events, tmin=0., tmax=1.500, baseline=None)

# Prepare data
X = np.array([mne.compute_covariance(
              meg_epochs[ii], method='oas')['data'][None]
              for ii in range(len(meg_epochs))])

y = emg_epochs.get_data().var(axis=2)[:, 0]  # target is EMG power

n_sub, n_fb, n_ch, _ = X.shape

# Define models
identity = ProjIdentitySpace()
lw = ProjLWSpace(shrink=shrink)
commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=reg)

logdiag = LogDiag()
naivevec = NaiveVec(method='upper')
riemanngeo = Riemann(n_fb=n_fb, metric=metric)
riemannwass = RiemannSnp(n_fb=n_fb, rank=n_compo)

sc = StandardScaler()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))

scorings = ['r2']
nfolds = [5]
n_compo = 151
allscores = dict()
import warnings
warnings.filterwarnings("ignore")

# make 5 groups
splits = np.array_split(np.arange(len(y)), 5)
groups = np.zeros(len(y), dtype=np.int)
for val, inds in enumerate(splits):
    groups[inds] = val

cv = GroupShuffleSplit(n_splits=10, train_size=.8, test_size=.2)
n_compos = [10, 20, 30, 35, 40, 45, 50, 55, 75]
for scoring in scorings:
    for n_compo in n_compos:

        # print('Pipeline: Identity + LogDiag')
        # pipe = make_pipeline(identity, logdiag, sc, ridge)
        # scores = cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
        #                          scoring=scoring,
        #                          groups=groups,
        #                          error_score=np.nan)
        # print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
        # scores_sensor_logdiag = scores

        # print('Pipeline: SPoC + LogDiag')
        # scores_sup_logdiag = []
        # for shrink in shrinks:
        #     spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo,
        #                          reg=reg)
        #     pipe = make_pipeline(spoc, logdiag, sc, ridge)
        #     scores = cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
        #                              scoring=scoring,
        #                              groups=groups,
        #                              error_score=np.nan)
        #     print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
        #     scores_sup_logdiag.append(scores)
        # scores_sup_logdiag = scores_sup_logdiag[np.argmin(np.mean(
        #     scores_sup_logdiag, axis=1))]

        # print('Pipeline: LW + RiemannGeo')
        # pipe = make_pipeline(lw, riemanngeo, sc, ridge)
        # scores = cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
        #                          scoring=scoring,
        #                          groups=groups,
        #                          error_score=np.nan)
        # print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
        # scores_sensor_riemanngeo = scores

        # print('Pipeline: Identity + NaiveVec')
        # pipe = make_pipeline(identity, naivevec, sc, ridge)
        # scores = cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
        #                          scoring=scoring,
        #                          groups=groups,
        #                          error_score=np.nan)
        # print("CV score: %.2f +/- %.2f" % (np.mean(scores), np.std(scores)))
        # scores_sensor_naivevec = scores
        # print('Pipeline: CommonEucl + RiemannGeo')
        scores_unsup_riemanngeo = []
        print("n components", n_compo)
        for reg in regs:
            commoneucl = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
            pipe = make_pipeline(commoneucl, riemanngeo, sc, ridge)
            scores = cross_val_score(pipe, X=X, y=y, cv=cv, n_jobs=n_jobs,
                                     scoring=scoring,
                                     groups=groups,
                                     error_score=np.nan)
            scores_unsup_riemanngeo.append(scores)
        scores_unsup_riemanngeo = scores_unsup_riemanngeo[np.argmax(
            np.mean(scores_unsup_riemanngeo, axis=1))]
        print("CV score: %.2f +/- %.2f" % (np.mean(scores_unsup_riemanngeo),
                                           np.std(scores_unsup_riemanngeo)))

  
        # break

        allscores[scoring+'-'+str(nfold)+'folds'] = {
            'sensor_logdiag': np.array(scores_sensor_logdiag),
            'sup_logdiag': np.array(scores_sup_logdiag),
            'sensor_naivevec': np.array(scores_sensor_naivevec),
            'sensor_riemanngeo': np.array(scores_sensor_riemanngeo),
            'unsup_riemanngeo': np.array(scores_unsup_riemanngeo)}

# np.save(op.join(cfg.path_outputs,
#                 'all_scores_models_fieldtrip_spoc_test.npy'), allscores)
