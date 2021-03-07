import glob
import os.path as op
import numpy as np
import mne
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

input_path = "/storage/inria/agramfor/camcan_derivatives"

bands = [
    'alpha',
    'beta_high',
    'beta_low',
    'delta',
    'gamma_high',
    'gamma_lo',
    'gamma_mid',
    'low',
    'theta'
]

# assemble matrixes
data = list()
for band in bands:
    data.append(
        pd.read_hdf(
            op.join(input_path, f'mne_source_power_diag-{band}.h5'),
            'mne_power_diag'))

data = pd.concat(data, axis=1)
subjects = data.index.values
# use subjects we used in previous nips submission
new_subjects = ['CC510256', 'CC520197', 'CC610051', 'CC121795',
                'CC410182']
            
mask = ~np.in1d(subjects, new_subjects)
subjects = subjects[mask]

X = data.values[mask]

participants_fname = op.join(cfg.derivative_path, "participants.csv")
participants = pd.read_csv(participants_fname)
y = participants.set_index('Observations').age.loc[subjects].values

seed = 42
n_splits = 10
n_jobs = 1
model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 5, 100)))

cv_split = ShuffleSplit(test_size=.1, n_splits=100, random_state=seed)

scores = -cross_val_score(model, X=X, y=y, cv=cv_split, n_jobs=n_jobs,
                          scoring='neg_mean_absolute_error')

scores_mne = {'mne_shuffle_split': np.array(scores)}

cv_rep = RepeatedKFold(n_splits=10, n_repeats=10)
scores = -cross_val_score(model, X=X, y=y, cv=cv_rep, n_jobs=n_jobs,
                          scoring='neg_mean_absolute_error')

escores_mne['mne_rep_cv'] = np.array(scores)

np.save(op.join(input_path, 'scores_mag_models_mne_intervals.npy'),
        scores_mne)
np.save(op.join(input_path, 'scores_mag_models_mne_intervals_subjects.npy'),
        subjects)
