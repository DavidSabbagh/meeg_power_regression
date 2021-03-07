- **plot_figure_meg_results.r**: generate Fig. 4 comparing MAE of different models

> Input: *PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects.npy, scores_mag_models_mne_common.npy*

> Output:*$PATH_FIGURES/fig1_meg_data.png* (Fig. 4 of paper)

- **plot_figure_meg_results_full.r**
> Input: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects_interval_shuffle-split.npy*, *scores_mag_models_mne_intervals.npy*

> Output: $PATH_FIGURES/fig1_meg_data_full.png*

- **plot_figure_meg_results_full_intervals.r**
> Input: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects_interval_shuffle-split.npy*, *scores_mag_models_mne_intervals.npy*

> Output: $PATH_FIGURES/fig1_meg_data_full_intervals.png*

- **plot_figure_meg_results_intervals.r**
> Input: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects_interval_rep-kfold.npy*, *scores_mag_models_mne_intervals.npy*

> Output: *$PATH_FIGURESbo/fig1_meg_data_intervals.png*
>
- **compute_scores_models_nips.py**: model sensor space

> input: *path_outputs/covs_allch_oas.h5, info_allch.npy, participants.csv*

> output *path_outputs/all_scores_mag_models_nips.npy*

- **plot_scores_models_mnecommonsubjects.py**
> Input: *$PATH_OUTPUTS/all_scores_mag_models_mnecommonsubjects.npy*, *scores_mag_models_mne_common.npy*

> Output: *$PATH_OUTPUTS/plot_MAE_mag_models_mnecommonsubjects.png*

- **compute_scores_models_mne_connectivity.py**: model sensor space

> input: *path_outputs/covs_allch_oas.h5, info_allch.npy, scores_mag_models_mne_subjects.npy, GGT_mne.h5, participants.csv*

> output *path_outputs/all_scores_mag_models_mne_connectivity.npy*

- **plot_scores_models_mne_connectivity.py**
> Input: $PATH_OUTPUTS/all_scores_mag_models_mne_connectivity.npy*

> Output: *$PATH_OUTPUTS/plot_models_mne_connectivity.png*



