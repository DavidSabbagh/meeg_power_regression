import glob
import os.path as op
import numpy as np
import mne
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

input_path = "/storage/inria/agramfor/camcan_derivatives"

fname_pattern = "{subject}_cov_mne_{band}.h5"

fnames = glob.glob(op.join(input_path, fname_pattern.format(
    subject="*", band="*")))
fnames.sort()


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
for fname in fnames:
    subject = fname.split("/")[-1].split("_")[0]
    assert sum(subject in ff for ff in fnames) == len(bands)
    result = mne.externals.h5io.read_hdf5(fname)
    res = {"subject": result["subject"],
           "fmin": result["fmin"],
           "fmax": result["fmax"],
           "band": result["band"]}
    for ii, label in enumerate(result["label_names"]):
        res[label] = result["power"][ii][0]
    data.append(res)
data = pd.DataFrame(data)

names = result["label_names"]
X = np.concatenate([data.query(f"band == '{bb}'")[names].values
                    for bb in bands], 1)

subjects = data.query("fmin == 8")["subject"]

participants_fname = "/storage/local/camcan/data/participants.csv"
participants = pd.read_csv(participants_fname)
y = participants.set_index('Observations').age.loc[subjects].values

seed = 42
n_splits = 10
n_jobs = 5
model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 5, 100)))
cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

scores = -cross_val_score(model, X=X, y=y, cv=cv, n_jobs=n_jobs,
                          scoring='neg_mean_absolute_error')

scores_mne = {'mne': np.array(scores)}

path_outputs = op.join("/storage/local/camcan/outputs")
np.save(op.join(path_outputs, 'scores_mag_models_mne.npy'), scores_mne)
np.save(op.join(path_outputs, 'scores_mag_models_mne_subjects.npy'), subjects)

# save model average:
model.fit(X2, y2)

model_coefs = {
    "X": X2,
    "y": y2,
    "ridge_coef": model.steps[1][1].coef_
}

np.save(op.join(path_outputs, 'mne_ridge_model_coefs.npy'), model_coefs)


# rerun with reduced selection of subjects
subjects2 = np.load("/storage/local/camcan/outputs/"
                    "scores_mag_models_common_subjects.npy")

data2 = data[data.subject.isin(subjects2)]
X2 = np.concatenate([data2.query(f"band == '{bb}'")[names].values
                    for bb in bands], 1)
y2 = participants.set_index('Observations').age.loc[subjects2].values

model2 = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 5, 100)))
cv2 = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

scores2 = -cross_val_score(model2, X=X2, y=y2, cv=cv2, n_jobs=n_jobs,
                          scoring='neg_mean_absolute_error')

scores_mne2 = {'mne': np.array(scores)}

path_outputs = op.join("/storage/local/camcan/outputs")
np.save(op.join(path_outputs, 'scores_mag_models_mne_common.npy'), scores_mne2)

np.save(op.join(path_outputs, 'features_mag_models_mne_common.npy'), X2)
