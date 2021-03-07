import os.path as op
import glob

path_outputs = op.join('./outputs')  # path to outputs

camcan_path = '/storage/store/data/camcan'
camcan_meg_path = op.join(camcan_path,
                          'camcan47/cc700/meg/pipeline/release004/')
camcan_meg_raw_path = op.join(camcan_meg_path,
                              'data/aamod_meg_get_fif_00001')
path_data = camcan_meg_raw_path
files_raw = sorted(glob.glob(op.join(path_data,
                   'CC??????/rest/rest_raw.fif')))

mne_camcan_freesurfer_path = (
    '/storage/store/data/camcan-mne/freesurfer')

derivative_path = ('/storage/inria/agramfor/camcan_derivatives')
# derivative_path contains participants.csv

path_maxfilter_info = op.join(derivative_path, 'maxfilter')
# copied from cfg.camcan_meg_path+"data_nomovecomp/aamod_meg_maxfilt_00001" but additionaly contains sss_params coming from??
