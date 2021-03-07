import os
import os.path as op

import numpy as np

import mne
from joblib import Parallel, delayed

import config as cfg


def _get_subjects(trans_set):
    trans = 'trans-%s' % trans_set
    found = os.listdir(op.join(cfg.derivative_path, trans))
    if trans_set == 'halifax':
        subjects = [sub[4:4 + 8] for sub in found]
    elif trans_set == 'krieger':
        subjects = ['CC' + sub[:6] for sub in found]
    print("found", len(subjects), "coregistrations")
    return subjects, [op.join(cfg.derivative_path, trans, ff) for ff in found]


subjects, trans = _get_subjects(trans_set='krieger')
trans_map = dict(zip(subjects, trans))


def _compute_GGT(subject, kind):

    # compute source space
    src = mne.setup_source_space(subject, spacing='oct6', add_dist=False,
                                 subjects_dir=cfg.mne_camcan_freesurfer_path)
    trans = trans_map[subject]
    bem = cfg.mne_camcan_freesurfer_path + \
        "/%s/bem/%s-meg-bem.fif" % (subject, subject)

    # compute handle MEG data
    fname = op.join(
        cfg.camcan_meg_raw_path,
        subject, kind, '%s_raw.fif' % kind)

    raw = mne.io.read_raw_fif(fname)
    mne.channels.fix_mag_coil_types(raw.info)

    event_length = 5.
    raw_length = raw.times[-1]
    events = mne.make_fixed_length_events(
        raw,
        duration=event_length, start=0, stop=raw_length - event_length)

    # Compute the forward and inverse
    info = mne.Epochs(raw, events=events, tmin=0, tmax=event_length,
                      baseline=None, reject=None, preload=False,
                      decim=10).info
    fwd = mne.make_forward_solution(info, trans, src, bem)
    leadfield = fwd['sol']['data']
    return {'ggt': np.dot(leadfield, leadfield.T)}


def _run_all(subject, kind='rest'):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_GGT(subject, kind)
    except Exception as err:
        error = repr(err)
        print(error)
    out = dict(subject=subject, kind=kind, error=error)
    out.update(result)
    return out


out = Parallel(n_jobs=40)(
    delayed(_run_all)(subject=subject, kind='rest')
    for subject in subjects)

out_fname = op.join(
    cfg.path_data, 'GGT_mne.h5')
mne.externals.h5io.write_hdf5(out_fname, out, overwrite=True)
