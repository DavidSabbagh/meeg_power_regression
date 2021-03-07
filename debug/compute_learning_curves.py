import os.path as op

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
import mne

import config as cfg
from library.spfiltering import ProjCommonSpace
from library.featuring import Riemann

meg = 'mag'
n_compo = 71
scale = 1e22
reg = 1e-7
metric = 'wasserstein'
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

common = ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg)
riemann = Riemann(n_fb=n_fb, metric=metric)
sc = StandardScaler()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

pipe = make_pipeline(common, riemann, sc, ridge)
train_sizes = np.linspace(0.1, 1, 5)
train_sizes, train_scores, test_scores = learning_curve(
                    pipe, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
                    scoring='neg_mean_absolute_error')

scores = {'train_sizes': train_sizes,
          'train_scores': train_scores,
          'test_scores': test_scores}
np.save(op.join(cfg.path_outputs,
        'all_scores_learning_curves.npy'), scores)
