library(tidyverse)
library(reticulate)
library(ggbeeswarm)
source('config.r')
np <- import("numpy")

data <- np$load(
  "./outputs/all_scores_mag_models_mnecommonsubjects_interval_rep-kfold.npy",
  # "all_scores_mag_models_mnecommonsubjects.npy",
  allow_pickle = T)[[1]] %>%
  as.data.frame()

data["mne"] <- np$load(
  # "scores_mag_models_mne_common.npy",
  "./outputs/scores_mag_models_mne_intervals.npy",
  allow_pickle = T)[[1]]["mne_shuffle_split"] %>%
  as.data.frame()

data <- data[, (names(data) %in% c("sensor_logdiag",
                                   "sup_logdiag",
                                   "sup_riemannwass",
                                   "sup_riemanngeo",
                                   "mne"))]

data_long <- data %>% gather(key = "estimator", value = "score")
# move to long format
data_long$estimator <- factor(data_long$estimator)

# set distance types
est_types <- c(
  # "dummy",
  "log-diag",
  "log-diag",
  # "log-diag",
  # "log-diag",
  "Wasserstein",
  # "Wasserstein",
  # "Wasserstein",
  # "Wasserstein",
  # "geometric",
  "geometric",
  # "geometric",
  # "geometric",
  "MNE"
)

# categorical colors based on: https://jfly.uni-koeln.de/color/
color_cats <- c(
  "#000000",
  "#009D79",# blueish green
  "#E36C2F",  #vermillon
  "#EEA535",  # orange
  "#0072B2" #blue
)

# beef up long data
data_long$est_type <- rep(est_types, each = 100)  %>%
  factor(., levels = c("dummy", "log-diag", "Wasserstein", "geometric",
                       "MNE"))
data_long$fold <- rep(1:100, times = length(est_types))
  
# prepare properly sorted x labels
sort_idx <-  apply(data, 2, mean) %>% order()
levels_est <- c(
  # "dummy",
  "identity",
  # "random",
  # "unsupervised",
  "supervised",
  # "identity",
  # "random[r]",
  # "unsupervised[r]",
  # "SPoC[r]",
  # "identity",
  # "random",
  "identity",
  "unsupervised",
  "biophysics"
)[sort_idx]

ggplot(data = data_long %>% subset(estimator != "dummy"),
       mapping = aes(y = score, x = reorder(estimator, score))) +
  geom_beeswarm(alpha = 0.4, aes(color = est_type), size = 3.5) +
  geom_boxplot(mapping = aes(fill = est_type, color = est_type),
               alpha = 0.6,
               outlier.fill = NA, outlier.colour = NA) +
  theme_minimal() + 
  labs(y = "mean absolute error (years)", x = NULL) +
  theme(text = element_text(family = "Helvetica", size = 18),
        legend.position = "top", legend.text = element_text( size = 18)) +
  coord_flip() +
  scale_fill_manual(values = color_cats[2:5], name = NULL) +
  scale_color_manual(values = color_cats[2:5], name = NULL) +
  scale_x_discrete(labels = parse(text = levels_est))

ggsave("./figures/fig1_meg_data_intervals.png", width = 8, height = 6, dpi = 300)
ggsave("./figures/fig1_meg_data_intervals.pdf", width = 8, height = 6, dpi = 300)
