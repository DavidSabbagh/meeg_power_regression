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

print('Running individual A experiment...')

# Parameters
n_matrices = 100  # Number of matrices
n_channels = 5   # Number of channels
n_sources = 2  # Number of sources
distance_A_id = 1.0  # Parameter 'mu': distance from A to Id
f_powers = 'log'  # link function between the y and the source powers
direction_A = None  # random direction_A
rng = 4
sigma = 0.  # Noise level in y
noises_A = np.logspace(-2.5, 0, 10)  # noise in Ai -2.5 -.2
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
    'dummy': make_pipeline(identity, logdiag, sc, dummy),
    'upper': make_pipeline(identity, upper, sc, ridge),
    'diag': make_pipeline(identity, logdiag, sc, ridge),
    'spoc': make_pipeline(spoc, logdiag, sc, ridge),
    'riemann': make_pipeline(identity, riemann, sc, ridge)
}

# Run experiments
results = np.zeros((len(pipelines), len(noises_A)))
for j, noise_A in enumerate(noises_A):
    X, y = generate_covariances(n_matrices, n_channels, n_sources,
                                sigma=sigma, distance_A_id=distance_A_id,
                                f_p=f_powers, direction_A=direction_A,
                                noise_A=noise_A, rng=rng)
    X = X[:, None, :, :]
    for i, (name, pipeline) in enumerate(pipelines.items()):
        print('noise_A = {}, {} method'.format(noise_A, name))
        sc = cross_val_score(pipeline, X, y, scoring=scoring,
                             cv=10, n_jobs=3, error_score=np.nan)
        results[i, j] = - np.mean(sc)

# save results
np.savetxt(op.join(cfg.path_outputs,
           'simuls/individual_spatial/scores.csv'),
           results, delimiter=',')
np.savetxt(op.join(cfg.path_outputs,
           'simuls/individual_spatial/names.csv'),
           np.array(list(pipelines)), fmt='%s')
np.savetxt(op.join(cfg.path_outputs,
           'simuls/individual_spatial/noises_A.csv'),
           noises_A)

# draft plot
#  f, ax = plt.subplots(figsize=(4, 3))
#  results /= results[0]
#  for i, name in enumerate(list(pipelines)):
#      if name != 'Chance level':
#          ls = None
#      else:
#          ls = '--'
#      ax.plot(noises_A, results[i],
#              label=name,
#              linewidth=3,
#              linestyle='--' if name == 'dummy' else None)
#  ax.set_xlabel('individual noise')
#  plt.grid()
#  ax.set_ylabel('Normalized M.A.E.')
#  ax.hlines(0, noises_A[0], noises_A[-1], label=r'Perfect',
#            color='k', linestyle='--', linewidth=3)
#  ax.legend(loc='lower right')
#  ax.set_xscale('log')
#  plt.show()
