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
                 'all_scores_learning_curves.npy')).item()
train_sizes = scores['train_sizes']
train_scores = scores['train_scores']
test_scores = scores['test_scores']

train_mean = - np.mean(train_scores, axis=1)
train_std = - np.std(train_scores, axis=1)
test_mean = - np.mean(test_scores, axis=1)
test_std = - np.std(test_scores, axis=1)

ax.plot(train_sizes, train_mean, 'b--', lw=2, label="Training score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.1)
ax.plot(train_sizes, test_mean, 'b-', label="CV score")
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                alpha=0.1, color="b")

#  ax.set_xticks(train_sizes)
ax.set_xlabel("Number of training examples")
ax.set_ylabel("MAE", rotation=0)
#  ax.set_title('Learning Curve (SpatialFilter + Riemann)')
ax.legend()
plt.tight_layout()
plt.savefig(op.join(cfg.path_outputs, 'plot_MAE_learning_curves.png'),
            dpi=300)
