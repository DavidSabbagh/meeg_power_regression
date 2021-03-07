import mne
import os.path as op
from autoreject import get_rejection_threshold

subject = 'CC110037'
kind = 'rest'
raw = mne.io.read_raw_fif(
        '/storage/local/camcan/data/'
        '{0:s}/{1:s}/{2:s}_raw.fif'.format(subject, kind, kind))
mne.channels.fix_mag_coil_types(raw.info)
raw.info['bads'] = ['MEG1031', 'MEG1111', 'MEG1941']

sss_params_dir = '/storage/local/camcan/maxfilter'
cal = op.join(sss_params_dir, 'sss_params', 'sss_cal.dat')
ctc = op.join(sss_params_dir, 'sss_params', 'ct_sparse.fif')
raw = mne.preprocessing.maxwell_filter(
      raw, calibration=cal,
      cross_talk=ctc,
      st_duration=10.,
      st_correlation=.98,
      destination=None,
      coord_frame='head')

eog_epochs = mne.preprocessing.create_eog_epochs(raw)
if len(eog_epochs) >= 5:
    reject_eog = get_rejection_threshold(eog_epochs, decim=8)
    del reject_eog['eog']  # we don't want to reject eog based on eog.
else:
    reject_eog = None

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
if len(ecg_epochs) >= 5:
    reject_ecg = get_rejection_threshold(ecg_epochs, decim=8)
    # here we want the eog.
else:
    reject_ecg = None

if reject_eog is None:
    reject_eog = {k: v for k, v in
                  reject_ecg.items() if k != 'eog'}

proj_eog, _ = mne.preprocessing.compute_proj_eog(
    raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

proj_ecg, _ = mne.preprocessing.compute_proj_ecg(
    raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

raw.add_proj(proj_eog)
raw.add_proj(proj_ecg)
