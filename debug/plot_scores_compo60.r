library(tidyverse)

data <- read_csv("./scores_mag_compo60.csv")

# remove unwanted columns
exclude_cols <- c("X1", "dummy", "commonwass_riemanngeo", "commonwass_logdiag",
                  "lw_logdiag")
data <- data[, !names(data) %in% exclude_cols]

data_long <- data %>% gather(key = "estimator", value = "score")
# move to long format
data_long$estimator <- factor(data_long$estimator)

# set distance types
est_types <- c(
  "log(diag)",
  "log(diag)",
  # "log_diag",
  # "log_diag",
  "log(diag)",
  "log(diag)",
  "geodesic",
  "geodesic",
  # "riemann_geo",
  "geodesic",
  "geodesic",
  "Wasserstein"
)

# categorical colors based on: https://jfly.uni-koeln.de/color/
color_cats <- c(
  "#009D79",# blueish green
  "#EEA535",  # orange
  # "#56B4E9",# sky blue
  # "#F0E442", #yellow
  # "#0072B2", #blue
  "#E36C2F"  #vermillon
  # "#CC79A7" #violet
)

# beef up long data
data_long$est_type <- rep(est_types, each = 10)  %>%
  factor(., levels = c("log(diag)", "Wasserstein", "geodesic"))
data_long$fold <- rep(1:10, times = length(est_types))
  
# prepare properly sorted x labels
sort_idx <-  apply(data, 2, mean) %>% order()
levels_est <- c(
  "id[sensor]",
  "random[sensor]",
  # "lw",
  # "commonwass",
  "common[sensor]",
  "SPoC[sensor]",
  "random[Riemann]",
  "lw[Riemann]",
  # "commonwass",
  "common[Riemann]",
  "SPoC[Riemann]",
  "id[Riemann]"
)[sort_idx]

ggplot(data = data_long,
       mapping = aes(y = score, x = reorder(estimator, score))) +
  geom_jitter(alpha = 0.5, aes(color = est_type), size = 2) +
  geom_boxplot(mapping = aes(fill = est_type), alpha = 0.7) +
  theme_minimal() + 
  labs(y = "mean absolute error (years)", x = "estimator") +
  theme(text = element_text(family = "Helvetica", size = 18),
        legend.position = "top") +
  coord_flip() +
  scale_fill_manual(values = color_cats, name = NULL) +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_x_discrete(labels = parse(text = levels_est))

ggsave("fig_real_data_performance.png", width = 8, height = 6, dpi = 300)
ggsave("fig_real_data_performance.pdf", width = 8, height = 6, dpi = 300)
