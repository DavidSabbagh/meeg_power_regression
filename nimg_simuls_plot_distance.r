library(tidyr)
library(ggplot2)

source("./config.r")
annotate_text_size <- 7
##############################################################################
# Experiment 1 data
da_exp <- read.csv("./outputs/simuls/synth_da/scores.csv", nrows = 5, header = F)
for (ii in c(2, 3, 4, 5, 1)){
  da_exp[ii,] <- da_exp[ii,] / da_exp[1,]
}
da_exp <- da_exp[2:5,] %>%
  gather(key = "estimator", value = "score")

estimator <- c("upper", "logdiag", "SPoC", "Riemann") %>%
  factor(, levels = c("upper", "logdiag", "SPoC", "Riemann"))
da_exp$estimator <- rep(estimator, times = 10)

distance <- read.csv("./outputs/simuls/synth_da/distance_a.csv", header = F)
da_exp$xaxis <- rep(distance[["V1"]], each = 4)

color_cats <- c(
  "#56B4E9",# sky blue
  "#009D79",# blueish green
  "#E36C2F",  #vermillon
  "#EEA535"  # orange
  # "#F0E442", #yellow
  # "#0072B2", #blue
  # "#CC79A7" #violet
)

ggplot(
  data = da_exp,
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator, fill = estimator)) +
  geom_hline(yintercept = 1., color = "black", linetype = "dotted",
             size = 1) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 4, shape = 21) +
  my_theme +
  scale_x_continuous(breaks = seq(0, 3, 0.5)) +
  scale_y_continuous(limits = c(0, 1.10), breaks = seq(0, 1, 0.25)) +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  annotate(geom = "text", x = 0.5, y = 1.1, label = "chance level",
           size = annotate_text_size) +
  labs(x = expression(mu),
       y = "Normalized MAE") +
  theme(legend.position = "top", legend.title = element_text(size = 16))

fname <- "./outputs/fig_1a_distance_loglinear"
ggsave(paste0(fname, ".png"), width = 5, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"), width = 5, height = 5, dpi = 300,
  useDingbats = F)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))

##############################################################################
# Experiment 2 data

da_exp2 <- read.csv(
  "./outputs/simuls/synth_da/scores_powers.csv", nrows = 5, header = F)
for (ii in c(2, 3, 4, 5, 1)){
  da_exp2[ii,] <- da_exp2[ii,] / da_exp2[1,]
}
da_exp2 <- da_exp2[2:5,] %>%
  gather(key = "estimator", value = "score")

estimator <- c("upper", "diag", "SPoC", "Riemann") %>%
  factor(, levels = c("upper", "diag", "SPoC", "Riemann"))
da_exp2$estimator <- rep(estimator, times = 10)

distance <- read.csv("./outputs/simuls//synth_da/distance_a.csv", header = F)
da_exp2$xaxis <- rep(distance[["V1"]], each = 4)

color_cats <- c(
  "#56B4E9",# sky blue
  "#009D79",# blueish green
  "#E36C2F",  #vermillon
  "#EEA535"  # orange
  # "#F0E442", #yellow
  # "#0072B2", #blue
  # "#CC79A7" #violet
)

ggplot(
  data = da_exp2,
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator, fill = estimator)) +
  geom_hline(yintercept = 1., color = "black", linetype = "dotted",
             size = 1) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 4, shape = 21) +
    my_theme +
  scale_x_continuous(breaks = seq(0, 3, 0.5)) +
  scale_y_continuous(limits = c(0, 1.10), breaks = seq(0, 1, 0.25)) +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  annotate(geom = "text", x = 0.5, y = 1.1, label = "chance level",
           size = annotate_text_size) +
  labs(x = expression(mu),
       y = "Normalized MAE") +
  theme(legend.position = "top", legend.title = element_text(size = 16))

fname <- "./outputs/fig_1a_distance_linear"
ggsave(paste0(fname, ".png"), width = 5, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"), width = 5, height = 5, dpi = 300,
  useDingbats = F)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))

