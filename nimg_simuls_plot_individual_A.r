library(tidyr)
library(ggplot2)

source("./config.r")

annotate_text_size <- 7
##############################################################################
# Experiment 1 data

noise_exp <- read.csv("./outputs/simuls/individual_spatial/scores.csv",
                      nrows = 5, header = F)
for  (ii in c(2, 3, 4, 5, 1)){
  noise_exp[ii,] <- noise_exp[ii,] / noise_exp[1,]
}
noise_exp <- noise_exp[2:5,] %>%
  gather(key = "estimator", value = "score")

estimator_levels <- c("upper", "logdiag", "SPoC", "Riemann")
estimator <- estimator_levels %>% factor(levels = estimator_levels)
noise_exp$estimator <- rep(estimator, times = 10)

noises <- read.csv("./outputs/simuls/individual_spatial/noises_A.csv", header = F)
noise_exp$xaxis <- rep(noises[["V1"]], each = 4)

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
  data = noise_exp %>% subset(estimator != "chance"),
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator)) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 4, shape = 21) +
  my_theme +
  scale_y_continuous(limits = c(0, 1.10), breaks = seq(0, 1, 0.25)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  scale_color_manual(values = color_cats, name = NULL) +
  geom_hline(yintercept = 1, color = "black", linetype = "dotted",
             size = 1) +
  annotate(geom = "text", x = 0.01, y = 1.05, label = "chance level",
           size = annotate_text_size) +
  labs(x = expression(sigma),
       y = "Normalized MAE") +
  theme(text = element_text(family = "Helvetica", size = 18),
        legend.position = "top", legend.title = element_text(size = 16))

fname <- "./outputs/fig_1c_individual_A_loglinear"
ggsave(paste0(fname, ".png"), width = 5, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"), width = 5, height = 5, dpi = 300,
  useDingbats = F)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))

##############################################################################
# Experiment 2 data

noise_exp <- read.csv("./outputs/simuls/individual_spatial/scores_powers.csv",
                      nrows = 5, header = F)
for  (ii in c(2, 3, 4, 5, 1)){
  noise_exp[ii,] <- noise_exp[ii,] / noise_exp[1,]
}
noise_exp <- noise_exp[2:5,] %>%
  gather(key = "estimator", value = "score")

estimator_levels <- c("upper", "diag", "SPoC", "Riemann")
estimator <- estimator_levels %>% factor(levels = estimator_levels)
noise_exp$estimator <- rep(estimator, times = 10)

noises <- read.csv("./outputs/simuls/individual_spatial/noises_A.csv", header = F)
noise_exp$xaxis <- rep(noises[["V1"]], each = 4)

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
  data = noise_exp %>% subset(estimator != "chance"),
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator)) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 4, shape = 21) +
  my_theme +
  scale_y_continuous(limits = c(0, 1.10), breaks = seq(0, 1, 0.25)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  scale_color_manual(values = color_cats, name = NULL) +
  geom_hline(yintercept = 1, color = "black", linetype = "dotted",
             size = 1) +
  annotate(geom = "text", x = 0.01, y = 1.05, label = "chance level",
           size = annotate_text_size) +
  labs(x = expression(sigma),
       y = "Normalized MAE") +
  theme(legend.position = "top", legend.title = element_text(size = 16))

fname <- "./outputs/fig_1c_individual_A_linear"
ggsave(paste0(fname, ".png"), width = 5, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"), width = 5, height = 5, dpi = 300,
  useDingbats = F)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))
