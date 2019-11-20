import os.path as op
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from library.simuls import generate_covariances
from library.spfiltering import ProjIdentitySpace, ProjSPoCSpace
from library.featuring import Diag, LogDiag, NaiveVec, Riemann

import config as cfg

print('Running distance experiment...')

# Parameters
n_matrices = 100  # Number of matrices
n_channels = 5   # Number of channels
n_sources = 2  # Number of sources
distances = np.linspace(0.001, 3, 10)  # Parameter 'mu': distance from A to Id
f_powers = 'linear'  # link function between the y and the source powers
direction_A = None  # random direction_1
rng = 4
sigma = 0.  # Noise level in y
noise_A = 0.
scoring = 'neg_mean_absolute_error'

# define spatial filters
identity = ProjIdentitySpace()
spoc = ProjSPoCSpace(n_compo=n_channels, scale='auto', reg=0, shrink=0)

# define featuring
upper = NaiveVec(method='upper')
diag = Diag()
logdiag = LogDiag()
riemann = Riemann(n_fb=1, metric='riemann')

sc = StandardScaler()

# define algo
dummy = DummyRegressor()
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100), scoring=scoring)

# define models
pipelines = {
    'dummy': make_pipeline(identity, diag, sc, dummy),
    'upper': make_pipeline(identity, upper, sc, ridge),
    'diag': make_pipeline(identity, diag, sc, ridge),
    'spoc': make_pipeline(spoc, diag, sc, ridge),
    'riemann': make_pipeline(identity, riemann, sc, ridge)
}

# Run experiments
results = np.zeros((len(pipelines), len(distances)))
for j, distance_A_id in enumerate(distances):
    X, y = generate_covariances(n_matrices, n_channels, n_sources,
                                sigma=sigma, distance_A_id=distance_A_id,
                                f_p=f_powers, direction_A=direction_A,
                                rng=rng)
    X = X[:, None, :, :]
    for i, (name, pipeline) in enumerate(pipelines.items()):
        print('distance = {}, {} method'.format(distance_A_id, name))
        sc = cross_val_score(pipeline, X, y, scoring=scoring,
                             cv=10, n_jobs=3, error_score=np.nan)
        results[i, j] = - np.mean(sc)

# save results
np.savetxt(op.join(cfg.path_outputs, 'simuls/synth_da/scores_powers.csv'),
           results, delimiter=',')
np.savetxt(op.join(cfg.path_outputs, 'simuls/synth_da/names_powers.csv'),
           np.array(list(pipelines)), fmt='%s')
np.savetxt(op.join(cfg.path_outputs, 'simuls/synth_da/distance_a_powers.csv'),
           distances)

# draft plot to check
#  f, ax = plt.subplots(figsize=(4, 3))
#  results /= results[0]
#  for i, name in enumerate(list(pipelines)):
#      ax.plot(distances, results[i],
#              label=name,
#              linewidth=3,
#              linestyle='--' if name == 'dummy' else None)
#  ax.set_xlabel('distance')
#  plt.grid()
#  ax.set_ylabel('Normalized M.A.E.')
#  ax.hlines(0, distances[0], distances[-1], label=r'Perfect',
#            color='k', linestyle='--', linewidth=3)
#  ax.legend(loc='lower right')
#  f.tight_layout()
#  plt.show()
