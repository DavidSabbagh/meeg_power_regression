#%%
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CV = 'shuffle-split'
CV = 'rep-kfold'

#%%
models = np.load(
    "./outputs/all_scores_mag_models_mnecommonsubjects_interval"
    f"_{CV}.npy", allow_pickle=True).item()

models['mne'] = np.load(
    "./outputs/scores_mag_models_mne_intervals.npy",
    allow_pickle=True
).item()['mne_shuffle_split']


#%%
pairs = (
    ('sensor_logdiag', 'sup_logdiag'),
    ('sup_logdiag', 'sensor_riemannwass'),
    ('sensor_riemannwass', 'unsup_riemanngeo'),
    ('unsup_riemanngeo', 'mne')
)

pair_diffs = list()
for second, leader in pairs:
    # second = 'sensor_logdiag'
    diff = pd.DataFrame({'diff': models[leader] - models[second]})
    prop_better = np.mean(models[leader] - models[second] < 0)
    diff['method'] = f'{leader} < {second} ({prop_better} %)'
    pair_diffs.append(diff)
pair_diffs = pd.concat(pair_diffs, axis=0)
#%%
sns.violinplot(x='diff', y='method', data=pair_diffs, cut=0)
ax = plt.gca()
ax.set_xlabel('Difference in MAE [years]')
plt.savefig(f'./figures/paired_diff_{CV}.png', dpi=300,
            bbox_inches='tight')

#%%

#%%
baseline_diffs = list()
for _, leader in pairs:
    second = 'sensor_logdiag'
    diff = pd.DataFrame({'diff': models[leader] - models[second]})
    prop_better = np.mean(models[leader] - models[second] < 0)
    diff['method'] = f'{leader} < {second} ({prop_better} %)'
    baseline_diffs.append(diff)
baseline_diffs = pd.concat(baseline_diffs, axis=0)
sns.violinplot(x='diff', y='method', data=baseline_diffs, cut=0)
ax = plt.gca()
ax.set_xlabel('Difference in MAE [years]')
plt.savefig(
    f'./figures/baseline_diff_{CV}.png', dpi=300,
    bbox_inches='tight')

#%%
