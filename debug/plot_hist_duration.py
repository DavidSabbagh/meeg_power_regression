import mne
import config_drago as cfg
import matplotlib.pyplot as plt

durations = []
for ff in cfg.files_raw:
    raw = mne.io.read_raw_fif(ff)
    duration = raw.n_times / raw.info['sfreq']/60
    durations.append(duration)

plt.hist(durations, bins=100)
plt.ylim(0, 10)
plt.show()
