import os.path as op
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import config as cfg

scores = np.load(op.join(cfg.path_outputs,
    'all_scores_mag_models_mnecommonsubjects.npy')).item()
scores.pop('dummy')

font = {'family': 'normal',
        'size': 16}
matplotlib.rc('font', **font)

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.boxplot(scores.values(), labels=scores.keys(), vert=False)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()

fig.savefig(op.join(cfg.path_outputs,
            'plot_MAE_mag_models_mnecommonsubjects.png'), dpi=300)
