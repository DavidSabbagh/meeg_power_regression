PYTHON ?= python
R ?= Rscript

all: clean simu fieldtrip camcan_source camcan_sensor error_decompo preproc tuh 

#################################################################
clean: clean-pyc clean-cache
	find . -name "Rplots.pdf" | xargs rm -f

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

#################################################################
simu: simu_compute simu_plot

simu_compute:
	$(PYTHON) nimg_simuls_compute_distance_linearpower.py
	$(PYTHON) nimg_simuls_compute_distance_loglinearpower.py
	$(PYTHON) nimg_simuls_compute_individual_A_linearpower.py
	$(PYTHON) nimg_simuls_compute_individual_A_loglinearpower.py
	$(PYTHON) nimg_simuls_compute_snr_linearpower.py
	$(PYTHON) nimg_simuls_compute_snr_loglinearpower.py

simu_plot:
	$(R) nimg_simuls_plot_distance.r
	$(R) nimg_simuls_plot_individual_A.r
	$(R) nimg_simuls_plot_snr.r

#################################################################
fieldtrip: fieldtrip_compute fieldtrip_plot

fieldtrip_compute: 
	$(PYTHON) compute_scores_models_fieldtrip_spoc_example.py
	
fieldtrip_plot:
	$(R) plot_figure_fieldtrip_results_intervals.r

#################################################################
camcan_source: camcan_source_compute camcan_source_plot

camcan_source_compute:
	$(PYTHON) compute_covs.py
	$(PYTHON) compute_cov_inverse_mne.py
	$(PYTHON) compute_scores_models_camcan_source.py

camcan_source_plot:
	$(R) plot_figure_camcan_source_model_comp.r

#################################################################
camcan_sensor: camcan_sensor_compute camcan_sensor_plot

camcan_sensor_compute:
	$(PYTHON) rewrite_outputs.py
	$(PYTHON) run_mne_projection_ridge.py
	$(PYTHON) run_mne_projection_ridge_interval.py
	$(PYTHON) compute_scores_models_camcan.py
	# $(PYTHON) compute_scores_models_mnesubjects.py
	# $(PYTHON) compute_scores_models_mnesubjects_intervals.py
	# $(PYTHON) compute_scores_models.py

camcan_sensor_plot:
	$(R) plot_figure_camcan_model_comp.r
	$(R) plot_figure_camcan_model_comp_noMNE.r

#################################################################
error_decompo: error_decompo_compute error_decompo_plot

error_decompo_compute: 
	$(PYTHON) compute_ggt.py
	$(PYTHON) compute_camcan_error_decomposition.py
	

error_decompo_plot:
	$(R) plot_figure_error_decomposition.r

#################################################################
preproc: preproc_compute preproc_plot

preproc_compute: 
	$(PYTHON) compute_covs_preproc_impact.py
	$(PYTHON) compute_camcan_preproc_impact.py
	$(PYTHON) compute_scores_models_mnesubjects_preproc_impact.py

preproc_plot:
	$(R) plot_figure_preproc_impact.r

#################################################################
tuh: tuh_compute tuh_plot

tuh_compute:
	$(PYTHON) compute_covs_tuh.py
	$(PYTHON) compute_scores_models_tuh.py

tuh_plot:
	$(R) plot_figure_tuh_model_comp.r
