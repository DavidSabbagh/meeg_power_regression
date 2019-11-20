library(reticulate)
library(cowplot)
library(ggbeeswarm)
source("config.r")

np <- import("numpy")
data <- np$load(
  paste0("./outputs/camacan_preproc_impact.npy"),
  allow_pickle = T)

# unpack
data <- data[[1]]
# make baseline appear in all both SSP-ER and SSS series
# concat first enetry [1]
data <- c(data[1], data)

# sort by SSP/SSP series
sorter <- c(c(1, 3, 4, 5, 6, 7), c(2, 8, 9, 10, 11, 12))
data <- data[sorter]

preproc_experiments <- names(data)

exp_seq <- seq_len(length(preproc_experiments))

exp_structure <- do.call(rbind, lapply(exp_seq, function(idx){
  x <- preproc_experiments[[idx]]
  x <- gsub("ssp_", "ssp-", x)
  x <- gsub("do_ar", "do-ar", x)
  xlist <- strsplit(x, "_")
  keys <- sapply(xlist, function(y) {
    substring(y, 1, nchar(y) - 2)
  })
 flags <- sapply(xlist, function(y) {
    as.logical(as.numeric(substring(y, nchar(y), nchar(y))))
  })
  out <- data.frame(t(flags))
  names(out) <- keys
  out[['id']] <- factor(idx)
  return(out)
}))

exp_structure$sub_id <- factor(rep(1:6, times = 2))
exp_structure$series <- factor(rep(c("SSP", "SSS"), each = 6))

n_folds <- length(data[[11]][[1]])

model_names <- c('riemann_53', 'riemann', 'spoc', 'spoc_67', 'log-diag')

data_results_long <- do.call(rbind, lapply(exp_seq, function(idx) {
    x <- data[[idx]]
    out <- data.frame(
        score = do.call("c", sapply(model_names, function(name) x[name])),
        estimator = factor(rep(model_names, each = n_folds)))
    out[['id']] <- factor(idx)
    out[['sub_id']] <- exp_structure$sub_id[idx]
    out[['series']] <- exp_structure$series[idx]
    return(out)}
))

data_plot <- merge(exp_structure, data_results_long, by = "id")

my_cl <- function(x){
  out <- data.frame(
    y = mean(x),
    ymin = quantile(x, probs = c(0.05)),
    ymax = quantile(x, probs = c(0.95)))
  return(out)
}

my_colors <- setNames(
  with(color_cats, c(blue, vermillon)),
  c('SSS', 'SSP')
)

proc_label <- c('env', 'eog', 'ecg', 'eo/cg', 'rej')

raw_means <- aggregate(score ~ estimator,
    data = subset(data_results_long, sub_id == 1),
    FUN = mean)
raw_means$label <- 'raw'

# raw_means$series <- c("SSP", "SSS")

# data_results_long$raw_score <- factor(data_results_long$raw_score)

data_plot <- subset(
  data_results_long, sub_id != 1 & !estimator %in% c("riemann", "spoc"))
data_plot$estimator <- factor(
  data_plot$estimator, levels = unique(data_plot$estimator))

data_plot$raw_score <- raw_means$score[3]
data_plot[data_plot$estimator == 'log-diag',]$raw_score <- raw_means$score[1]
data_plot[data_plot$estimator == 'spoc_67',]$raw_score <- raw_means$score[5]

estimator_labs <- setNames(
  c("Riemann[53]", "SPoC[67]", "diag"),
  c("riemann_53", "spoc_67", "log-diag")
)

fig_preproc <- ggplot(
    data = data_plot,
    mapping = aes(x = sub_id, y = score, group = series, color = series)) +
    geom_hline(aes(yintercept = raw_score), linetype = 'dashed') +
    geom_text(
      data = subset(raw_means, !estimator %in% c("riemann", "spoc")),
    aes(y = score, x = 1.2, label = label),
      color = color_cats[['black']], vjust = -0.5, size = 4,
      inherit.aes = F) +
    geom_beeswarm(alpha = 0.2, size = 1.5,
                  dodge.width = 0.5) +
    stat_summary(geom = 'line', fun.y = mean,
                 position = position_dodge(width = 0.5), size = 1) +
    stat_summary(geom= 'point', fun.y = mean,
                 position = position_dodge(width = 0.5),
                 shape = 21, fill = 'white', size = 4) +
    scale_y_continuous(breaks = 6:14) +
    coord_cartesian(ylim = c(6, 14)) +
    facet_wrap(~estimator, nrow = 1,
               labeller = as_labeller(estimator_labs, label_parsed)) +
    scale_x_discrete(labels = proc_label) + 
    scale_color_manual(
      breaks = names(my_colors),
      values = my_colors,
      name = NULL) +
    ylab("MAE") +
    xlab("Preprocessing steps") +
    my_theme + 
    theme(legend.position = c(0.8, 0.07)) +
    guides(color = guide_legend(nrow=1))
fig_preproc

fname <- './outputs/preproc_impact'
ggsave(paste0(fname, '.pdf'), 
       plot = fig_preproc, width = 10, height = 4,
       useDingbats = FALSE)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))

ggsave(paste0(fname, '.png'),
       plot =  fig_preproc, width = 10, height = 4)
