import os.path as op
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg

sns.set_style('darkgrid')
sns.set_context('notebook')
sns.despine(trim=True)
plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

scores = np.load(op.join(cfg.path_outputs,
                 'all_scores_duration_mag_scale22_reg7_ridge.npy')).item()
durations = scores['durations']
scores_model = scores['scores_model']
scores_base = scores['scores_base']

mean = scores_model.mean(axis=1)
std = scores_model.std(axis=1)
ax.plot(durations, mean, 'b-', lw=2, label='Model (SpatialFilter + Riemann)')
#  ax.fill_between(durations, mean - std, mean+std, alpha=0.1)
#  ax.hlines(mean[-1], xmin=1, xmax=9, colors='b', linestyle='dashed')

mean = scores_base.mean(axis=1)
std = scores_base.std(axis=1)
ax.plot(durations, mean, 'k--', lw=2, label='Baseline (Identity + LogDiag)')
#  ax.fill_between(durations, mean - std, mean+std, alpha=0.1)
#  ax.hlines(mean[-1], xmin=1, xmax=9, colors='g', linestyle='dashed')

ax.set_xticks(durations)
ax.set_xlabel('Duration of MEG recording (mns)')
ax.set_ylabel('MAE', rotation=0)
#  ax.set_title('Prediction error of age from CAMCAN MEG recordings\n'
#               'n=572-640, meg=mag, feat/compo=9, ridgeCV, KFolds=10\n'
#               'Common(compo=71, scale=1e22, reg=1e-7)')
ax.legend()
plt.tight_layout()
fig.savefig(op.join(cfg.path_outputs, 'plot_MAE_mag_models_duration.png'),
            dpi=300)
