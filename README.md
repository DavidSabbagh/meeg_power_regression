Regression with covariance matrices
===================================

This is the Python code for the NIMG 2020 article
**Predictive regression modeling with MEG/EEG: from source power to signals
and cognitive states**

The content of the `Makefile` describes the scripts execution order to generate the figures of the paper

- *make clean*: clean the repo

- *make simu*: compute & plot results of simulations (Fig.2)

- *make fieldtrip*: compute & plot results of FieldTrip experiment (Fig.3)

- *make camcan_sensor*: compute & plot results of Cam-CAN experiment in sensor space (Fig.4)

- *make camcan_source*: compute & plot results of Cam-CAN experiment in source space (Fig.5)

- *make error_decompo*: compute & plot results of error decomposition experiment (Fig.6)

- *make preproc*: compute & plot results of pre-processing impact (Fig.7) 

- *make tuh*: compute & plot results of TUH EEG experiment (Fig.8)

Dependencies
------------

 - numpy >= 1.15
 - scipy >= 1.12
 - matplotlib >= 3.0
 - scikit-learn >= 0.20
 - h5py
 - pyriemann (https://github.com/alexandrebarachant/pyRiemann)


Configuration file
-----------

- **config.py** defines global PATH variables 
Please change *PATH_OUTPUTS* to output directory and *PATH_DIR* to Cam-CAN input data.

- **config.r** defines global variables for plotting 


Libraries
---------

- **/library/preprocessing.py** contains the code used to preprocess raw data from CamCAN

- **/library/spfiltering.py** contains the functions to implement spatial filtering of the covariance matrices

- **/library/featuring.py** contains all the functions to vectorize the covariance matrices 

- **/library/simuls**: contains the function to generate covariance matrices following the generative model of the paper

Debug
----------

The `debug` directory contains old scripts used to debug our experiments but also some scripts that were needed for old versions of current experiments so potentially useful to you. As such, please look into this directory if below scripts do not work 

Simulations
-------------

- **nimg_simuls_compute_xx.py**: scripts generating MAE scores for the 3
  simulations of the paper

> Input: None

> Output: *$PATH_OUTPUTS/simuls/xx/yy.csv*

- **nimg_simuls_plot_xx.r** are the corresponding plotting scripts (in R)

> Input: *$PATH_OUTPUTS/simuls/xx/yy.csv*

> Output: Fig. 2 of paper - fig1{a, b ,c}_{distance, snr, individual_A}_{linear, loglinear}.png

FieldTrip experiment
-------------------

- **compute_scores_models_fieldtrip_spoc_example.py**: scripts generating R2
  scores for the FielTrip experiment

> Input: None

> Output: *$PATH_OUTPUTS/all_scores_models_fieldtrip_spoc_{mae, r2}.npy, fieldtrip_component_scores.csv*

- **plot_figure_fieldtrip_results_intervals.r**: corresponding plotting script
  (in R)

> Input: $PATH_OUTPUTS/all_scores_models_fieldtrip_spoc_r2.npy, fieldtrip_component_scores.csv*

> Output: Fig. 3 of paper - *$PATH_OUTPUTS/fig_fieldtrip_component_selection.pdf, fig_fieldtrip_model_comp.pdf*

Cam-CAN source-space experiment
---------
To be run before the Cam-CAN sensor-space experiment because the subjects for which the source-space experiment worked is a subset of sensor-space experiment and we want to run both on the common subjects

- **compute_covs.py**: compute the sensor-space covariance  matrices for each Cam-CAN
  subject and frequency band

> Input: *$PATH_DATA/CC??????/rest/rest_raw.fif*

> Output: *$PATH_DERIVATIVES/covs_allch_oas.h5*

- **compute_cov_inverse_mne.py**: compute the source covariance matrices for each
  Cam-CAN subject and frequency band for the MNE model directly from raw signal and stored as upper triangle by this script.
  it does raw->preproch->epoching->filtering->epochs-to-source-vial-MNE->cov
  different approach than in NIPS/NIMG_before_rebuttal articles to compute MNE logdiag using GiSigmaGi.T 

> Input: *$PATH_MEG_RAW/CC??????/rest/CC??????_raw.fif, $PATH_CAMCAN_MEG/emptyroom/CC??????/emptyroom_CC??????.fif, $PATH_DERIVATIVES/trans-krieger/??????-ve_tasks-??????.fif (coregistration files), $CAMCAN_MEG_PATH/data_nomovecomp/aamod_meg_maxfilt_00001/CC??????/rest/mf2pt2_rest_raw.log (to parse bad channels from maxfilter info), $PATH_MNE_CAMCAN_FREESURFER/CC??????/bem/CC??????-meg-bem.fif (freesurfer BEM files)*

> Output: *$PATH_OUTPUTS/camcan/{subject}_cov_mne_{band}.h5, all_mne_source_power.h5*

- **compute_scores_models_camcan_source.py**: compute the scores of our models in source space
> Input: *$PATH_DERIVATIVES/covs_allch_oas.h5, all_mne_source_power.h5, participants.csv*

> Output: *$PATH_OUTPUTS/camcan_source_component_scores.csv, all_scores_models_camcan_source_{mae, neg_mean_absolute_error}_shuffle-split.npy*

- **plot_figure_camcan_source_model_comp.r**: plot models comparison figure in source space 
> Input: *$PATH_OUTPUTS/all_scores_models_camcan_source_mae_shuffle-split.npy, camcan_source_component_scores.csv*

> Ouput: Fig. 5 of paper - *$PATH_OUPUTS/fig_camcan_source_model_comp.png, fig_camcan_source_component_selection.png*


Cam-CAN sensor-space experiment
---------

- **rewrite_outputs.py**: all_mne_source_power.h5 is too big to fit into memory, so this script cut it in small pieces

> Input: *$PATH_OUTPUTS/all_mne_source_power.h5*

> Output: *$PATH_OUTPUTS/mne_source_power_diag-{band}.h5*

- **run_mne_projection_ridge.py**: compute MAE score of MNE model

> Input: *$PATH_OUTPUTS/camcan/{subject}_cov_mne_{band}.h5*, *participants.csv*

> Output: *$PATH_OUTPUTS/scores_mag_models_common_subjects.npy*,*scores_mag_models_mne.npy*, *scores_mag_models_mne_subjects.npy*, *mne_ridge_model_coefs.npy*, *scores_mag_models_mne_common.npy*, *features_mag_models_mne_common.npy*

- **run_mne_projection_ridge_interval.py**: compute MAE score of MNE model

> Input: *$PATH_OUTPUTS/mne_source_power_diag-{band}.h5*, *participants.csv*

> Output: *$PATH_OUTPUTS/scores_mag_models_mne_intervals.npy*, *scores_mag_models_mne_intervals_subjects.npy*

- **compute_scores_models_camcan.py**: model sensor space

> Input: *$PATH_OUTPUTS/covs_allch_oas.h5, info_allch.npy, scores_mag_models_mne_intervals_subjects.npy, participants.csv*

> Output *$PATH_OUTPUTS/camcan_component_scores.csv*, *all_scores_models_camcan_{mae, neg_mean_absolute_error}_shuffle-split.npy*

- **plot_figure_camcan_model_comp.r**: plot models comparison figure in sensor-space

> Input: *$PATH_OUTPUTS/camcan_component_scores.csv*, *all_scores_models_camcan_mae_shuffle-split.npy, *scores_mag_models_mne_intervals.npy*

> Ouput: Fig. 4 of paper - $PATH_OUTPUTS/fig_camcan_model_comp.png*, *fig_camcan_component_selection.png*

- **plot_figure_camcan_model_comp_noMNE.r**
> Input: *$PATH_OUPUTS/all_scores_models_camcan_mae_shuffle-split.npy*, *scores_mag_models_mne_intervals.npy*, *camcan_component_scores.csv*

> Output: Fig. 4 of paper - *$PATH_OUTPUTS/fig_camcan_model_comp.png*, *fig_camcan_component_selection.png*

Those files may or may not be needed:

- **compute_scores_models_mnesubjects.py**: compute MAE scores of different models

> Input: *$PATH_OUTPUTS/covs_allch_oas.h5, info_allch.npy, scores_mag_models_mne_subjects.npy, participants.csv*

> Output: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects.npy*


- **compute_scores_models_mnesubjects_intervals.py**: compute MAE scores of different models

> Input: *$PATH_OUTPUTS/covs_allch_oas.h5, info_allch.npy, scores_mag_models_mne_subjects.npy, participants.csv*

> Output: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects_interval_{shuffle-split, rep-kfold}.npy*


- **compute_scores_models.py**: model sensor space

> Input: *$PATH_OUTPUTS/covs_allch_oas.h5, info_allch.npy, scores_mag_models_mne_intervals_subjects.npy, participants.csv*

> Output *$PATH_OUTPUTS/all_scores_mag_models.npy*

Error decomposition
---------

- **compute_ggt.py**: Compute G.G.T where G is the leadfield

> Input: *$PATH_MNE_CAMCAN_FREESURFER/CC??????/bem/CC??????-meg-bem.fif* (freesurfer BEM files)

> Output: *$PATH_DATA/GGT_mne.h5*

- **compute_camcan_error_decomposition.py**: perform error decomposition described in paper

> Input: *$PATH_DATA/info_allch.npy, covs_allch_oas.h5, scores_mag_models_mne_subjects.npy, GGT_mne.h5, participants.csv, camcan_component_scores.csv*

> Output: *$PATH_OUTPUTS/all_scores_camcan_error_deomposition.npy*

- **plot_figure_error_decomposition.r**: plot error decomposition figure

> Input: *$PATH_OUTPUTS/all_scores_camcan_error_decomposition.npy, all_scores_models_camcan_mae_shuffle-split.npy, camcan_component_scores.csv*

> Output: Fig. 6 of paper - *$PATH_OUPUTS/fig_error_decomposition.png*


Preprocessing experiment
---------

- **compute_covs_preproc_impact.py**

> Input: *$PATH_DATA/CC??????/rest/rest_raw.fif*

> Output: *$PATH_DATA/covs_preproc_impact_{name}.h5*

- **compute_camcan_preproc_impact.py**

> Input: *$PATH_DATA/info_allch.npy, scores_mag_models_mne_subjects.npy, participants.csv, covs_preproc_impact_{name}.h5*

> Output: *$PATH_OUTPUTS/camcan_preproc_impact.npy*

- **compute_scores_models_mnesubjects_preproc_impact.py**

> Input: *$PATH_DATA/info_allch.npy, scores_mag_models_mne_subjects.npy, participants.csv, covs_preproc_impact_{name}.h5*

> Output: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects_preproc_impact.npy*

- **plot_figure_preproc_impact.r**

> Input: *$PATH_OUTPUTS/camcan_preproc_impact.npy*

> Output: Fig. 7 of paper - *$PATH_OUTPUTS/preproc_impact.png*


TUH experiment
----------------

- **compute_covs_tuh.py**

> Input: tuh_eeg_abnormal/v2.0.0 data directory

> Output: *$PATH_DERIVATIVES/covs_tuh_oas.h5*

- **compute_scores_models_tuh.py**

> Input: *$PATH_DERIVATIVES/covs_tuh_oas.h5*

> Output: *$PATH_OUTPUTS/tuh_component_scores.csv*, *all_scores_models_tuh_{score_name}_{cv_name}.npy*

- **plot_figure_tuh_model_comp.r**

> Input: *$PATH_OUTPUTS/tuh_component_scores.csv, all_scores_models_tuh_mae_shuffle-split.npy*

> Output: Fig. 8 of paper - *$PATH_OUTPUTS/fig_tuh_model_comp.png, fig_tuh_component_selection.png*
