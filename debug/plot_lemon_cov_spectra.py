import os.path as op
import glob
import numpy as np
import mne

import matplotlib.pyplot as plt

with open('./eeg_file_list.txt', 'r') as fid:
    file_list = fid.read().split('\n')[:-1]

subjects = list({ff.split('_')[0] for ff in file_list})

out_path = op.expanduser('~/study_data/LEMON/EEG_Preprocessed')

ref_info = mne.io.read_info("lemon-ref-info.fif")

cov_fnames = glob.glob("./derivatives/sub-??????-alpha-cov.fif")

cov_egivals = list()
for fname in cov_fnames:
    cov = mne.read_cov(fname)
    eigvals = np.linalg.svd(cov.data, full_matrices=True)[1]
    cov_egivals.append(eigvals)

cov_egivals = np.array(cov_egivals)
plt.semilogy(cov_egivals.T, color="black", alpha=0.2)
plt.semilogy(cov_egivals.T.mean(1), color="red", alpha=1)

# -> min rank is ~13.