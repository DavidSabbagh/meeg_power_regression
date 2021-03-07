import os.path as op
import numpy as np
import mne
from joblib import Parallel, delayed

with open('./eeg_file_list.txt', 'r') as fid:
    file_list = fid.read().split('\n')[:-1]

subjects = list({ff.split('_')[0] for ff in file_list})


out_path = op.expanduser('~/study_data/LEMON/EEG_Preprocessed')

report = mne.report.Report(title='single subject PSD')

ranks = list()
subs = list()


ref_info = mne.io.read_info("lemon-ref-info.fif")


def _interpolate_missing(raw, ref_info):
    missing_chs = [
        ch for ch in ref_info["ch_names"] if ch not in raw.ch_names]
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    picks_other = [ii for ii in range(len(raw.ch_names)) if ii not in
                   picks_eeg]
    other_chs = [raw.ch_names[po] for po in picks_other]
    n_channels = (len(picks_eeg) +
                  len(missing_chs) +
                  len(other_chs))

    assert len(ref_info['ch_names']) == n_channels
    existing_channels_index = [
        ii for ii, ch in enumerate(ref_info['ch_names']) if
        ch in raw.ch_names]

    shape = (n_channels, len(raw.times))
    data = raw.get_data()
    out_data = np.empty(shape, dtype=data.dtype)
    out_data[existing_channels_index] = data
    out = mne.io.RawArray(out_data, ref_info.copy())
    if raw.annotations is not None:
        out.set_annotations(raw.annotations)

    out.info['bads'] = missing_chs
    out.interpolate_bads(mode="fast")
    return out


freqs = [(0.1, 1.5, "low"),
         (1.5, 4.0, "delta"),
         (4.0, 8.0, "theta"),
         (8.0, 15.0, "alpha"),
         (15.0, 26.0, "beta_low"),
         (26.0, 35.0, "beta_high"),
         (35.0, 50.0, "gamma_lo"),
         (50.0, 74.0, "gamma_mid"),
         (76.0, 120.0, "gamma_high")]


def _compute_cov(subject):
 
    fname = subject + '_EC.set'
    raw = mne.io.read_raw_eeglab(op.join(out_path, fname))
    raw.annotations.duration[
        raw.annotations.description == 'boundary'] = 0.0

    raw.annotations.description[
        raw.annotations.description == 'boundary'] = "edge"

    raw = _interpolate_missing(raw, ref_info)
    for fmin, fmax, band in freqs:
        raw_f = raw.copy().filter(fmin, fmax)
        cov_f = mne.compute_raw_covariance(raw_f, method="oas")
        cov_f.save(f"./derivatives/{subject}-{band}-cov.fif")

out = Parallel(n_jobs=2)(delayed(_compute_cov)(sub) for sub in subjects)
