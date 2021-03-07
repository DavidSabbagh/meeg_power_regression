import glob
import re
import os.path as op

import numpy as np
from joblib import Parallel, delayed
import mne
from autoreject import get_rejection_threshold

import config as cfg
import library.preprocessing as pp


def rawfile_of(subject):
    rawfiles = [f for f in edfs if subject in f]
    return rawfiles[0]  # few subjects have multiple sessions


def age_of(subject, print_header=False):
    # TNTLFreiburg/brainfeatures/blob/master/brainfeatures/utils/file_util.py
    # teuniz.net/edfbrowser/edf%20format%20description.html
    fp = rawfile_of(subject)
    assert op.exists(fp), "file not found {}".format(fp)
    f = open(fp, 'rb')
    content = f.read(88)
    f.close()
    patient_id = content[8:88].decode('ascii')
    print(patient_id if print_header else None)
    [age] = re.findall("Age:(\\d+)", patient_id)
    return int(age)


def preprocess_raw(subject):
    raw_file = rawfile_of(subject)
    raw = mne.io.read_raw_edf(raw_file)
    raw.crop(tmin=60, tmax=540)  # 8mn of signal to be comparable with CAM-can
    raw.load_data().pick_channels(list(common_chs))
    raw.resample(250)  # max common sfreq

    # autoreject global (instead of clip at +-800uV proposed by Freiburg)
    duration = 3.
    events = mne.make_fixed_length_events(
            raw, id=3, start=0, duration=duration)
    epochs = mne.Epochs(raw, events, event_id=3, tmin=0, tmax=duration,
                        proj=False, baseline=None, reject=None)
    reject = get_rejection_threshold(epochs, decim=1)
    return raw, reject


def _compute_cov(subject):
    rawc, reject = preprocess_raw(subject)

    events = mne.make_fixed_length_events(
        rawc, id=3000, start=0, duration=pp.duration)
    epochs = mne.Epochs(
        rawc, events, event_id=3000, tmin=0, tmax=pp.duration, proj=True,
        baseline=None, reject=reject, preload=False, decim=1)
    epochs.drop_bad()
    clean_events = events[epochs.selection]

    #  picks = mne.pick_types(rawc.info, meg=False, eeg=True)
    covs = []
    for fb in pp.fbands:
        rf = rawc.copy().load_data().filter(fb[0], fb[1])
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=pp.duration,
            proj=True, baseline=None, reject=None, preload=False, decim=1,
            picks=None)
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    out = dict(subject=subject, n_events=len(events),
               n_events_good=len(clean_events),
               covs=np.array(covs), age=age_of(subject))
    return out


def _run_all(subject):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_cov(subject)
    except Exception as err:
        error = repr(err)
        print(error)
    out = dict(error=error)
    out.update(result)
    return out


# edf files are stored in root_dir/
# edf/{eval|train}/normal/01_tcp_ar/103/00010307/s001_2013_05_29/00010307_s001_t000.edf'
# '01_tcp_ar': the only type of channel configuration used in this corpus
# '103': header of patient id to make folders size reasonnable
# '00010307': patient id
# 's001_2013_01_09': session & archive date (~record date from EEG header)
# '00010194_s001_t001.edf': patient id, session number and token number of EEG
# segment
root_dir = ('/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/'
            'tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/')
edfs = list()
for tt in ['eval', 'train']:
    edfs += sorted(glob.glob(op.join(root_dir,
                             f'edf/{tt}/normal/01_tcp_ar/*/*/*/*.edf')))

subjects = sorted(list(set([edf.split('/')[-3] for edf in edfs])))

raw = mne.io.read_raw_edf(rawfile_of(subjects[0]))
common_chs = set(raw.info['ch_names'])
for sub in subjects[1:]:
    raw = mne.io.read_raw_edf(rawfile_of(sub))
    chs = set(raw.info['ch_names'])
    common_chs = common_chs.intersection(chs)
common_chs -= {'BURSTS', 'IBI', 'SUPPR',
               'EEG EKG1-REF', 'EEG T1-REF', 'EEG T2-REF'}

out = Parallel(n_jobs=40)(
    delayed(_run_all)(subject=subject)
    for subject in subjects)
fname_covs = op.join(cfg.derivative_path, 'covs_tuh_oas.h5')
mne.externals.h5io.write_hdf5(fname_covs, out, overwrite=True)

#  age = np.array([age_of(subject) for subject in subjects])
#  import matplotlib.pyplot as plt
#  plt.close('all')
#  plt.hist(age, bins=20)
#  plt.title('Age histogram of TUH Abnormal dataset')
#  plt.xlabel('Age')
#  plt.savefig(op.join(cfg.path_outputs, 'fig_tuh_hist_age.png'), dpi=300)
