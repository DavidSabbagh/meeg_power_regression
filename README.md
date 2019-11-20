Regression with covariance matrices
===================================

This is the Python code for the NIMG 2019 submitted article
**Predictive regression modeling with MEG/EEG: from source power to signals
and cognitive states**

- *make clean*: clean the repo

- *make simu*: compute & plot results of simulations (Fig.3)

- *make fieldtrip*: compute & plot results of FieldTrip experiment (Fig.4)

- *make camcan*: compute & plot results of Cam-CAN experiment (Fig.5)

- *make error_decompo*: compute & plot results of error decomposition
  experiment (Fig.6)

- *make preproc*: compute & plot results of pre-processing impact (Fig.7) 


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

- **config.py** defines global PATH variables *PATH_OUTPUTS* that should point
  to output directory and *PATH_DIR* to Cam-CAN input data


Libraries
---------

- **/library/preprocessing.py** contains the code used to preprocess raw data from CamCAN

- **/library/spfiltering.py** contains the functions to implement spatial filtering of the covariance matrices

- **/library/featuring.py** contains all the functions to vectorize the covariance matrices 

- **/library/simuls**: contains the function to  generate covariance matrices following the generative model of the paper

- **/library/utils.py** contains the other vectorization methods


Simulations
-------------

- **nimg_simuls_compute_xx.py**: scripts generating MAE scores for the 3
  simulations of the paper

> Input: None

> Output: PATH_OUTPUTS/simuls/xx/yy.csv

- **nimg_simuls_plot_xx.r** are the corresponding plotting scripts (in R)

> Input: PATH_OUTPUTS/simuls/xx/yy.csv

> Output: Fig. 3 of paper

FieldTrip experiment
-------------------

- **compute_scores_models_fieldtrip_spoc_example.py**: scripts generating R2
  scores for the FielTrip experiment

> Input: None

> Output: PATH_OUTPUTS/all_scores_models_fieldtrip_spoc_r2.npy, PATH_OUTPUTS/fieldtrip_component_scores.csv

- **plot_figure_fieldtrip_results_intervals.r**: corresponding plotting script
  (in R)

> Input: PATH_OUTPUTS/all_scores_models_fieldtrip_spoc_r2.npy, PATH_OUTPUTS/fieldtrip_component_scores.csv

> Output: PATH_OUTPUTS/fig_fieldtrip_component_selection.pdf, PATH_OUPTUTS/fig_fieldtrip_model_comp.pdf (Fig.4 of paper)


<!-- ## Neuroimage paper -->
<!--  -->
<!-- ### 2. Camcan data -->
<!--  -->
<!-- #### 2.1 Model comparison -->
<!--  -->
<!-- ##### features -->
<!--  -->
<!-- Sensor space: -->
<!--  -->
<!-- - "./compute_covs.py" -->
<!--  -->
<!-- MNE: -->
<!--  -->
<!-- - "./compute_cov_inverse_mne.py" -->
<!--  -->
<!-- ##### models -->
<!--  -->
<!-- Sensor space: -->
<!--  -->
<!-- - "./compute_scores_models_camcan.py" -->
<!--  -->
<!-- MNE: -->
<!--  -->
<!-- - "./run_mne_projection_ridge_interval.py" -->
<!--  -->
<!-- ##### Plotting -->
<!--  -->
<!-- - "./plot_figure_camcan_model_comp.r" -->
<!--  -->
<!-- #### 2.2 Error decomposition -->
<!--  -->
<!-- ##### Features -->
<!--  -->
<!-- - "./compute_covs.py" -->
<!--  -->
<!-- ##### Models -->
<!--  -->
<!-- - "./compute_camcan_error_decomposition.py" -->
<!--  -->
<!-- ##### Plotting -->
<!--  -->
<!-- - "./plot_figure_error_decomposition.r" -->
<!--  -->
<!-- #### 2.3 Preproc experiment -->
<!--  -->
<!-- ##### features -->
<!--  -->
<!-- - "./compute_covs_preproc_impact.py" -->
<!--  -->
<!-- ##### Models -->
<!--  -->
<!-- - "./compute_camcan_preproc_impact.py" -->
<!--  -->
<!-- ##### Plotting -->
<!--  -->
<!-- - "./compute_camcan_preproc_impact.py" -->
<!--  -->
<!-- MEG data: age prediction from Cam-CAN -->
<!-- --------- -->
<!-- - **compute_cov_inverse_mne.py**: compute the covariance matrices for each -->
<!--   Cam-CAN subject and frequency band for the MNE model -->
<!--  -->
<!-- > Input: None -->
<!--  -->
<!-- > Output: PATH_OUTPUTS/camcan/{subject}_cov_mne_{band}.h5 -->
<!--  -->
<!-- - **run_mne_projection_ridge.py**: compute MAE score of MNE model -->
<!--  -->
<!-- > Input: *PATH_OUTPUTS/camcan/{subject}_cov_mne_{band}.h5* -->
<!--  -->
<!-- > Output: *PATH_OUTPUTS/scores_mag_models_mne_common.npy* -->
<!--  -->
<!-- - **compute_covs.py**: compute the covariance  matrices for each Cam-CAN -->
<!--   subject and frequency band -->
<!--  -->
<!-- > Input: None -->
<!--  -->
<!-- > Output: *PATH_OUTPUTS/covs_allch_oas.h5* -->
<!--  -->
<!-- - **compute_scores_models_mnesubjects.py**: compute MAE scores of different models -->
<!--  -->
<!-- > Input: *PATH_OUTPUTS/covs_allch_oas.h5, info_allch.npy, scores_mag_models_mne_subjects.npy, participants.csv* -->
<!--  -->
<!-- > Output: *PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects.npy* -->
<!--  -->
<!-- - **plot_figure_meg_results.r**: generate Fig. 4 comparing MAE of different models -->
<!--  -->
<!-- > Input: *PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects.npy, -->
<!-- scores_mag_models_mne_common.npy* -->
<!--  -->
<!-- > Output: Fig. 4 of paper -->
<!--  -->
<!-- - **plot_patterns.py**: generate Fig. 5A of first SPoC pattern -->
<!--  -->
<!-- > Input: *PATH_OUTPUTS/covs_allch_oas.h5, scores_mag_models_mne_subjects.npy, -->
<!-- participants.csv* -->
<!--  -->
<!-- > Output: Fig. 5A of paper (plot_firstpattern_spoc.png, part of fig_4.png) -->
<!--  -->
