library(tidyverse)
library(reticulate)
library(ggbeeswarm)
source('config.r')
np <- import("numpy")

data <- np$load(
  "./outputs/all_scores_models_fieldtrip_spoc_test.npy",
# "all_scores_mag_models_mnecommonsubjects.npy",
  allow_pickle = T)[[1]]$'neg_mean_absolute_error-10folds' %>%
  as.data.frame()

data_ <- data[, (names(data) %in% c("sensor_logdiag",
                                   "sup_logdiag",
                                   "sensor_naivevec",
                                   "sensor_riemannwass",
                                   "sensor_riemanngeo"))]

data_long <- data_ %>% gather(key = "estimator", value = "score")
# move to long format
data_long$estimator <- factor(data_long$estimator)

# set distance types
est_types <- c(
  "log-diag",
  "log-diag",
  "euclidean",
  "Wasserstein",
  "geometric"
)

# categorical colors based on: https://jfly.uni-koeln.de/color/

my_color_cats <- with(
  color_cats,
  c(`blueish green`, `blueish green`, `sky blue`, vermillon, orange))

# beef up long data
data_long$est_type <- rep(est_types, each = 10) %>%
  factor(., levels = c("log-diag", "euclidean", "Wasserstein", "geometric"))
data_long$fold <- rep(1:10, times = length(est_types))

# prepare properly sorted x labels
sort_idx <- apply(data_, 2, mean) %>% order()
levels_est <- c(
  "identity",
  "supervised",
  "identity",
  "identity",
  "identity"
)[rev(sort_idx)]

ggplot(data = data_long %>% subset(estimator != "dummy"),
       mapping = aes(y = score, x = reorder(estimator, I(-score)))) +
  geom_beeswarm(
    priority = 'density',
    mapping = aes(color = est_type, size = 1 - score,
                  alpha = 1 - score),
    show.legend = T, cex = 0.65) +
  scale_size_continuous(range = c(0.5, 2)) +
  scale_alpha_continuous(range = c(0.4, 0.7)) +
  geom_boxplot(mapping = aes(fill = est_type, color = est_type),
               alpha = 0.4,
               outlier.fill = NA, outlier.colour = NA) +
  stat_summary(geom = 'text',
               mapping = aes(label  = sprintf("%1.2f",
                                              ..y..)),
               fun.y= mean, size = 3.2, show.legend = FALSE,
               position = position_nudge(x=-0.49)) +
  my_theme +
  labs(y = expression(MAE), x = NULL, parse = T) +
  guides(size = F, alpha = F) +
  theme(text = element_text(family = "Helvetica", size = 18),
        legend.position = "top", legend.text = element_text(size = 18)) +
  coord_flip() +
  scale_fill_manual(values = my_color_cats[2:6], name = NULL) +
  scale_color_manual(values = my_color_cats[2:6], name = NULL) +
  scale_x_discrete(labels = parse(text = levels_est))


fname <- "./figures_nimg_2019/fig_fieldtrip_model_comp_testMAE"
ggsave(paste0(fname, ".png"),
       width = 8, height = 4, dpi = 300)
ggsave(paste0(fname, ".pdf"),
       useDingbats = F,
       width = 8, height = 4, dpi = 300)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))
