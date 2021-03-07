import os.path as op

import numpy as np
from joblib import Parallel, delayed
import mne

import config_drago as cfg  # define dataset here
from library import preprocessing as pp


def _compute_cov(file_raw, duration):
    subject = pp.get_subject(file_raw)
    raw = mne.io.read_raw_fif(file_raw)
    raw.crop(tmax=duration*60)
    rawc, reject = pp.clean_raw(raw, subject)

    events = mne.make_fixed_length_events(
        rawc, id=3000, start=0, duration=pp.duration)
    epochs = mne.Epochs(
        rawc, events, event_id=3000, tmin=0, tmax=pp.duration, proj=True,
        baseline=None, reject=reject, preload=False, decim=1)
    epochs.drop_bad()
    clean_events = events[epochs.selection]

    picks = mne.pick_types(rawc.info, meg=True)
    covs = []
    for fb in pp.fbands:
        rf = rawc.copy().load_data().filter(fb[0], fb[1])
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=pp.duration,
            proj=True, baseline=None, reject=None, preload=False, decim=1,
            picks=picks)
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    out = dict(subject=subject, kind='rest', n_events=len(events),
               n_events_good=len(clean_events), covs=np.array(covs))
    return out


def _run_all(file_raw, duration=None):
    mne.utils.set_log_level('warning')
    subject = pp.get_subject(file_raw)
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_cov(file_raw, duration)
    except Exception as err:
        error = repr(err)
        print(error)
    out = dict(error=error)
    out.update(result)
    return out


durations = [1, 2, 4, 6]
durations = [6]
for duration in durations:
    print('Duration = %d mn' % duration)
    out = Parallel(n_jobs=10)(
        delayed(_run_all)(file_raw=file_raw, duration=duration)
        for file_raw in cfg.files_raw)
    fname_covs = op.join(cfg.path_outputs, 'covs_allch_oas_%d.h5' % duration)
    mne.externals.h5io.write_hdf5(fname_covs, out, overwrite=True)
