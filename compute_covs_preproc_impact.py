import os.path as op

import numpy as np
from joblib import Parallel, delayed
import mne

import config as cfg
from library import preprocessing as pp
from library.preprocessing import (
    _run_maxfilter, _get_global_reject_epochs, _get_global_reject_ssp)


def clean_raw(raw, subject, sss, ssp_er, ssp_eog, ssp_ecg, do_ar):
    mne.channels.fix_mag_coil_types(raw.info)
    if sss:
        raw = _run_maxfilter(raw, subject, 'rest')
    elif not ssp_er:
        raw.add_proj([], remove_existing=True)
    if ssp_eog:
        _compute_add_ssp_exg(raw, eog=True, ecg=False)
    if ssp_ecg:
        _compute_add_ssp_exg(raw, eog=False, ecg=True)

    if do_ar:
        reject = _get_global_reject_epochs(raw)
    else:
        reject = None
    return raw, reject


def _compute_add_ssp_exg(raw, eog=True, ecg=True):
    if eog or ecg:
        reject_eog, reject_ecg = _get_global_reject_ssp(raw)

    proj_eog, proj_ecg = None, None
    if 'eog' in raw and eog:
        proj_eog, _ = mne.preprocessing.compute_proj_eog(
            raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)
    if proj_eog is not None:
        raw.add_proj(proj_eog)

    if ecg:
        proj_ecg, _ = mne.preprocessing.compute_proj_ecg(
            raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)
    if proj_ecg is not None:
        raw.add_proj(proj_ecg)


def _compute_cov(file_raw, sss, ssp_er, ssp_eog, ssp_ecg, do_ar):
    subject = pp.get_subject(file_raw)
    raw = mne.io.read_raw_fif(file_raw)
    if DEBUG:
        raw.crop(0, 30)

    rawc, reject = clean_raw(
        raw, subject, sss=sss, ssp_er=ssp_er, ssp_eog=ssp_eog, ssp_ecg=ssp_ecg,
        do_ar=do_ar)

    events = mne.make_fixed_length_events(
        rawc, id=3000, start=0, duration=pp.duration)
    epochs = mne.Epochs(
        rawc, events, event_id=3000, tmin=0, tmax=pp.duration, proj=True,
        baseline=None, reject=reject, preload=False, decim=8 if DEBUG else 1)
    epochs.drop_bad()
    clean_events = events[epochs.selection]

    picks = mne.pick_types(rawc.info, meg=True)
    covs = []
    for fb in pp.fbands[:1 if DEBUG else len(pp.fbands)]:
        rf = rawc.copy().load_data().filter(fb[0], fb[1])
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=pp.duration,
            proj=True, baseline=None, reject=None, preload=False,
            decim=8 if DEBUG else 1,
            picks=picks)
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    out = dict(subject=subject, kind='rest', n_events=len(events),
               n_events_good=len(clean_events), covs=np.array(covs))
    return out


def _run_all(file_raw, sss, ssp_er, ssp_eog, ssp_ecg, do_ar):
    mne.utils.set_log_level('warning')
    subject = pp.get_subject(file_raw)
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_cov(file_raw, sss=sss, ssp_er=ssp_er, ssp_eog=ssp_eog,
                              ssp_ecg=ssp_ecg, do_ar=do_ar)
    except Exception as err:
        error = repr(err)
        print(error)
    out = dict(error=error)
    out.update(result)
    return out

params = [
    # sss
    dict(sss=False, ssp_er=False, ssp_ecg=False, ssp_eog=False, do_ar=False),
    dict(sss=True, ssp_er=False, ssp_ecg=False, ssp_eog=False, do_ar=False),
    dict(sss=True, ssp_er=False, ssp_ecg=True, ssp_eog=False, do_ar=False),
    dict(sss=True, ssp_er=False, ssp_ecg=False, ssp_eog=True, do_ar=False),
    dict(sss=True, ssp_er=False, ssp_ecg=True, ssp_eog=True, do_ar=False),
    dict(sss=True, ssp_er=False, ssp_ecg=True, ssp_eog=True, do_ar=True),
    # ssp empty room
    dict(sss=False, ssp_er=True, ssp_ecg=False, ssp_eog=False, do_ar=False),
    dict(sss=False, ssp_er=True, ssp_ecg=True, ssp_eog=False, do_ar=False),
    dict(sss=False, ssp_er=True, ssp_ecg=False, ssp_eog=True, do_ar=False),
    dict(sss=False, ssp_er=True, ssp_ecg=True, ssp_eog=True, do_ar=False),
    dict(sss=False, ssp_er=True, ssp_ecg=True, ssp_eog=True, do_ar=True)
]

subjects = cfg.files_raw
DEBUG = False
if DEBUG:
    subjects_ = subjects[:1]
else:
    subjects_ = subjects

for this_params in params:
    out = Parallel(n_jobs=20)(
        delayed(_run_all)(file_raw=file_raw, **this_params)
        for file_raw in subjects_)
    
    name = '_'.join([f"{k}-{int(v)}" for k, v in this_params.items()])
    fname_covs = op.join(cfg.path_data, f'covs_preproc_impact_{name}.h5')
    mne.externals.h5io.write_hdf5(fname_covs, out, overwrite=True)
