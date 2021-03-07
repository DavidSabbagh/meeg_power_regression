import os.path as op
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import config as cfg

font = {'family': 'normal',
        'size': 16}
matplotlib.rc('font', **font)

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

scores = np.load(op.join(
    cfg.path_outputs, 'all_scores_mag_scaleAuto_reg0_ridge.npy')).item()
compos = scores['compos']
shrinks = scores['shrinks']

methods = ['id_logdiag', 'common_logdiag', 'common_riemann', 'spoc_logdiag',
           'spoc_riemann', 'id_riemannsnp']
names = ['No | powers', 'Unsup.| powers',
         'Unsup. | $S_n^{++}$', 'Sup. | powers',
         'Sup. | $S_n^{++}$', 'No | $S_{n,p}^+$']
colors = ['k', 'b', 'b', 'r', 'r', 'k']
linestyles = ['--', '--', '-', '--', '-', '-']

for (method, name, color, linestyle) in zip(methods, names, colors,
                                            linestyles):
    score = scores[method]
    if 'spoc' in method:
        mean = score.mean(axis=2).min(axis=1)
        idx = score.mean(axis=2).argmin(axis=1)
        print('idx=', idx)
        std = score.std(axis=2)[np.arange(len(compos)), idx]
    elif method == 'id_logdiag':
        mean = score.mean(axis=0) * np.ones(len(compos))
        std = score.std(axis=0) * np.ones(len(compos))
    else:
        mean = score.mean(axis=1)
        std = score.std(axis=1)
    ax.plot(compos, mean, color+linestyle, label=name)

ax.set_xticks(compos)
xl = ax.set_xlabel('Number of spatial filters')
yl = ax.set_ylabel('MAE')
lgd = ax.legend(ncol=2)
plt.grid()
plt.ylim((8.0, 12.0))
plt.show()
fig.savefig(op.join(cfg.path_outputs,
            'plot_MAE_mag_scaleAuto_reg0_models.pdf'),
            bbox_extra_artists=[xl, yl, lgd],
            bbox_inches='tight')
