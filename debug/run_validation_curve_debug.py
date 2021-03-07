import mne

from autoreject import validation_curve, get_rejection_threshold  # noqa

import numpy as np  # noqa
param_range = np.linspace(40e-6, 200e-6, 30)

epochs = mne.read_epochs('data/camcan-debug-subject-CC210148-epo.fif')
peaks = np.ptp(X, axis=-1)
param_range = np.sort(peaks.max(1))

plt.plot(param_range, -test_scores.mean(1))

best_thresh = param_range[::5][np.argmin(-test_scores.mean(1))]

mean = -test_scores.mean(1) 
std = -test_scores.std(1)

plt.plot(param_range[::5], -test_scores.mean(1))

plt.figure()
plt.plot(param_range[::5], -test_scores.mean(1), color='red')
plt.plot(param_range[::5], -test_scores, color='blue')
plt.axvline(best_thresh * 1.05, color='black')

plt.fill_between(param_range[::5], mean + std, mean - std, alpha=.3)