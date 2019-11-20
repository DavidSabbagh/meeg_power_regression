library(tidyr)
library(magrittr)
library(reticulate)
library(cowplot)
library(ggbeeswarm)
source("config.r")

np <- import("numpy")
data <- np$load(
  "./outputs/all_scores_camcan_error_decomposition.npy",
# "all_scores_mag_models_mnecommonsubjects.npy",
  allow_pickle = T)

data_mod_comp <- np$load(
  "./outputs/all_scores_models_camcan_mae_shuffle-split.npy",
# "./outputs/all_scores_mag_models_mnecommonsubjects_interval_rep-kfold.npy",
# "all_scores_mag_models_mnecommonsubjects.npy",
  allow_pickle = T)[[1]] %>%
  as.data.frame()

data_comp_scores <- read.csv(
  "outputs/camcan_component_scores.csv",
  stringsAsFactor = T
)

data_comp_scores_long <- data.frame(
  score = c(data_comp_scores$spoc, data_comp_scores$riemann),
  estimator = factor(rep(c("SPoC", "Riemann"),
                         each = nrow(data_comp_scores))),
  n_components = rep(data_comp_scores$n_components, times = 2),
  fold_idx = factor(rep(data_comp_scores$fold_idx, times = 2))
)

agg_scores <- aggregate(cbind(spoc, riemann) ~ n_components,
                        data = data_comp_scores, FUN = mean)
dummy_mean <- mean(data_mod_comp$dummy)

n_folds <- length(data[[1]]$leadfield$spoc)
sim_names <- names(data[[1]])
n_sim_models <- length(sim_names)

model_names_ <- names(data[[1]]$full)
model_names <- model_names_[c(5, 3, 4, 2, 1)]
model_labels <- c(
  sprintf("Riemann[%d]", which.min(agg_scores$riemann)),
          "Riemann",
  sprintf("SPoC[%d]", which.min(agg_scores$spoc)),
          "SPoC", "diag")

n_models <- length(model_names)
n_rows <- n_folds * n_sim_models * n_models

get_long <- function() {
  scores <- c()
  estimator <- c()
  simulation <- c()
  for (simu in sim_names){
    for (model in model_names){
      print(model)
      scores <- c(scores, data[[1]][[simu]][[model]])
      estimator <- c(estimator, rep(model, n_folds))
      simulation <- c(simulation, rep(simu, n_folds))
    }
  }
  data.frame(
    score = scores,
    estimator = estimator,
    generator = simulation
)}

data_long <- get_long()
data_long$estimator <- factor(data_long$estimator, levels = model_names)

data_long$generator <- factor(data_long$generator, levels = sim_names)

print(
  aggregate(score~estimator*generator, data_long, mean))

data_diff <- do.call("rbind", by(
  data_long,
  list(data_long$estimator, data_long$generator),
  FUN = function(x){
    x_ref <- data_long[data_long$generator == "full" &
                       data_long$estimator == x$estimator[[1]],]
    data.frame(
      score = x$score - mean(x_ref$score),
      estimator = x$estimator,
      generator = x$generator)
}))
levels(data_diff$estimator)

my_colors <- setNames(
  with(color_cats, c(black, vermillon, orange)),
  c("full", "power", "leadfield")
)
my_labels <- setNames(
  c("full", "leadfield + power", "leadfield"),
  c("full", "power", "leadfield")
)

data_diff$score <- -data_diff$score

data_plot <- rbind(data_long, data_diff)
data_plot$mode <- 'absolute'
data_plot$mode[(nrow(data_plot) / 2 + 1):nrow(data_plot)] <- 'diff'
data_plot$mode <- factor(data_plot$mode)
data_plot$hline <- dummy_mean
data_plot$hline[data_plot$mode == 'diff'] <- 0

fun_breaks <- function(x) {
  if (min(x) > 0) c(6, 8, 10, 12, 14, 16) else c(-10, -8, -6, -4, -2, 0) 
}

data_plot$est_type <- gsub(
  sprintf("spoc_%d", which.min(agg_scores$spoc)),
  "spoc", data_plot$estimator)
data_plot$est_type <- gsub(
  sprintf("riemann_%d", which.min(agg_scores$riemann)),
  "riemann", data_plot$est_type)

model_labels_best <- model_labels[!model_names %in% c("riemann", "spoc")]

model_labels_shape <- model_labels[
  model_names %in% c("log-diag", "riemann", "spoc")]
model_names_shape <- model_names[
  model_names %in% c("log-diag", "riemann", "spoc")]

shape_labels <- setNames(
  model_labels_shape,
  model_names_shape
)

shape_values <- setNames(
  c(0, 1, 2),
  model_names_shape
)

annot_data <- data.frame(
  x = c(3, 5),
  y = c(dummy_mean, 0),
  label = c("predicting~bar(age)", "full[i]-bar(full)"),
  mode = c("absolute", "diff"))

fig_error <- ggplot(
    data = subset(data_plot, score < 18 & score > -10 &
                  !estimator %in% c("riemann", "spoc")),
    mapping = aes(y = score,
                  x = interaction(estimator, generator),
                  shape = est_type, 
                  color = generator,
                  fill = generator)) +
  geom_beeswarm(alpha=0.25, size = 2, cex = 1.1) +
  coord_flip() +
  geom_boxplot(alpha = 0.25, outlier.shape = NA, width = 0.9) +
  facet_wrap(.~mode,  scales = 'free_x') +
  scale_x_discrete(
    labels = parse(text = rep(model_labels_best, times = n_sim_models))) +
  guides(shape = guide_legend(title = 'Estimator')) +
  scale_color_manual(breaks = names(my_colors),
                     values = my_colors,
                     labels = my_labels,
                     name = 'Generator') +
  scale_fill_manual(breaks = names(my_colors),
                    values = my_colors,
                    labels = my_labels,
                    name = 'Generator') +
  scale_shape_manual(breaks = names(shape_values),
                     values = shape_values,
                     labels = shape_labels, 
                     name = 'Estimator') +
  my_theme +
  scale_y_continuous(breaks = fun_breaks) +
  geom_hline(aes(yintercept = hline),
             linetype = 'dashed',
             color = color_cats[["black"]]) +
  geom_text(
    data = annot_data,
  aes(y = y, x = x, label = label),
    color = color_cats[['black']], vjust = -0.5,
    angle = 270,
    inherit.aes = F,
    parse = T,
    size = annotate_text_size) +
  ylab("MAE") +
  xlab("Model") +
  theme(legend.position = c(0.6, 0.5))

fig_error

fname <- './outputs/fig_error_decomposition'
ggsave(paste0(fname, '.pdf'),
       plot = fig_error, width = 10, height = 4,
       useDingbats = FALSE)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))

ggsave(paste0(fname, '.png'),
       plot = fig_error, width = 10, height = 4)
