import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import config as cfg

scores = np.load(op.join(cfg.path_outputs,
                 'all_scores_mag_models_mne_connectivity.npy')).item()
scores_dummy = scores['dummy']
scores_id_logdiag = np.concatenate((scores_dummy[None, :],
                                   scores['id_logdiag']),
                                   axis=0)[::-1]
scores_unsup_riemanngeo = np.concatenate((scores_dummy[None, :],
                                         scores['unsup_riemanngeo']),
                                         axis=0)[::-1]

labels = ['Dummy',
          'Leadfield',
          'Leadfield + Powers',
          'Leadfield + Powers + Connectivity'][::-1]

plt.close('all')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4), sharex=True)
bplot1 = axes[0].boxplot(scores_id_logdiag.T, vert=False,
                         patch_artist=True, labels=labels)
axes[0].set_title('Model: Identity + Logdiag')

bplot2 = axes[1].boxplot(scores_unsup_riemanngeo.T, vert=False,
                         patch_artist=True)
axes[1].set_title('Model: Unsupervised + RiemannGeo')

colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.tight_layout()
plt.savefig(op.join(cfg.path_outputs, 'plot_models_mne_connectivity.png'),
            dpi=300)
