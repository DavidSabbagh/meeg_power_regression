import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import mne

import config_drago as cfg

info = np.load(op.join(cfg.path_data, 'info_allch.npy')).item()
picks = mne.pick_types(info, meg='mag')

fname = op.join(cfg.path_outputs, 'covs_allch_oas.h5')
data = mne.externals.h5io.read_hdf5(fname)  # (sub, fb, ch, ch)

subjects = [d['subject'] for d in data if 'subject' in d]
covs = [d['covs'][:, picks][:, :, picks] for d in data if 'subject' in d]
covs = np.array(covs)  # (sub,fb,chan,chan)

ranks = [] 
for sub in range(len(subjects)):
    cov = mne.Covariance(covs[sub][4], np.array(info['ch_names'])[picks],
                         [], [], 1)
    ranks.append(mne.compute_rank(cov, info=info)['mag'])
plt.figure()
plt.hist(ranks)
