import mne
import os.path as op
from autoreject import get_rejection_threshold

subject = 'CC210148'
kind = 'rest'
fname = ('/storage/local/camcan/data/'
         '{0:s}/{1:s}/{2:s}_raw.fif').format(subject, kind, kind)
raw = mne.io.read_raw_fif(fname)
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
    del reject_eog['eog']
else:
    reject_eog = None

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
if len(ecg_epochs) >= 5:
    reject_ecg = get_rejection_threshold(ecg_epochs, decim=8)
else:
    reject_eog = None

if reject_eog is None:
    reject_eog = reject_ecg
if reject_ecg is None:
    reject_ecg = reject_eog

proj_eog = mne.preprocessing.compute_proj_eog(
    raw, average=True, reject=reject_eog, n_mag=1, n_grad=1, n_eeg=1)

proj_ecg = mne.preprocessing.compute_proj_ecg(
    raw, average=True, reject=reject_ecg, n_mag=1, n_grad=1, n_eeg=1)

raw.add_proj(proj_eog[0])
raw.add_proj(proj_ecg[0])

# XXX this was 3 sec before
duration = 3.
events = mne.make_fixed_length_events(
    raw, id=3000, start=0, duration=duration)

epochs = mne.Epochs(
    raw, events, event_id=3000, tmin=0, tmax=duration, proj=False,
    baseline=None, reject=None)
# epochs.apply_proj()
epochs.load_data()
epochs.pick_types(meg=True)
# XXX decim was 8 before
reject = get_rejection_threshold(epochs, decim=8)

stop = raw.times[-1]
duration = 30.
overlap = 8.
stop = raw.times[-1]
events = mne.make_fixed_length_events(
    raw, id=3000, start=0, duration=overlap,
    stop=stop - duration)

epochs = mne.Epochs(
    raw, events, event_id=3000, tmin=0, tmax=duration, proj=True,
    baseline=None, reject={k: v * 1 for k, v in reject.items()},
    preload=True, decim=1)

picks = mne.pick_types(raw.info, meg=True)
psd, freqs = mne.time_frequency.psd_welch(
    epochs, fmin=0, fmax=150, n_fft=4096,  # ~12 seconds
    n_overlap=512,
    picks=picks)
