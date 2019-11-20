PYTHON ?= python
R ?= Rscript

all: clean simu plot_simu camcan plot_camcan

clean: clean-pyc clean-cache
	find . -name "Rplots.pdf" | xargs rm -f

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

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

fieldtrip: fieldtrip_compute fieldtrip_plot

fieldtrip_compute: 
	$(PYTHON) compute_scores_models_fieldtrip_spoc_example.py
	
fieldtrip_plot:
	$(R) plot_figure_fieldtrip_results_intervals.r

camcan: camcan_compute camcan_plot

camcan_compute:
	# $(PYTHON) compute_covs.py
	# $(PYTHON) compute_cov_inverse_mne.py
	# $(PYTHON) compute_scores_models_mnesubjects.py
	# $(PYTHON) run_mne_projection_ridge.py

camcan_plot:
	$(R) plot_figure_camcan_model_comp.r

error_decompo: error_decompo_compute error_decompo_plot

error_decompo_compute: 

error_decompo_plot:
	$(R) plot_figure_error_decomposition.r

preproc: preproc_compute preproc_plot

preproc_compute: 

preproc_plot:
	$(R) plot_figure_preproc_impact.r
