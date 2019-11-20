library(tidyr)
library(ggplot2)

source("./config.r")

annotate_text_size <- 7
##############################################################################
# Experiment 2 data

snr_exp <- read.csv("./outputs/simuls/synth_snr/scores.csv", nrows = 5, header = F)
for (ii in c(2, 3, 4, 5, 1)){
  snr_exp[ii,] <- snr_exp[ii,] / snr_exp[1,]
}

snr_exp <- snr_exp %>%
  gather(key = "estimator", value = "score")

estimator2 <- c("chance", "upper", "logdiag", "SPoC", "Riemann") %>%
  factor(, levels = c("chance", "upper", "logdiag", "SPoC", "Riemann"))

snr_exp$estimator <- rep(estimator2, times = 10)

sigmas <- read.csv("./outputs/simuls/synth_snr/sigmas.csv", header = F)
snr_exp$xaxis <- rep(sigmas[["V1"]], each = 5)

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
  data = snr_exp %>% subset(estimator != "chance"),
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator)) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 4, shape = 21) +
  my_theme +
  scale_y_continuous(limits = c(0, 1.10), breaks = seq(0, 1, 0.25)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  scale_color_manual(values = color_cats, name = NULL) +
  # labs(x = TeX("distance between A and $I_p$"),
  #      y = "normalized M.A.E.") +
  geom_hline(yintercept = 1, color = "black", linetype = "dotted",
             size = 1) +
  annotate(geom = "text", x = 0.05, y = 1.04, label = "chance level",
           size = annotate_text_size) +
  labs(x = expression(sigma),
       y = "Normalized MAE") +
  theme(legend.position = "top", legend.title = element_text(size = 16))

fname <- "./outputs/fig_1b_snr_loglinear"
ggsave(paste0(fname, ".png"), width = 5, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"), width = 5, height = 5, dpi = 300,
  useDingbats = F)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))
##############################################################################
# Experiment 2.2 data

snr_exp <- read.csv("./outputs/simuls/synth_snr/scores_powers.csv", nrows = 5, header = F)
for (ii in c(2, 3, 4, 5, 1)){
  snr_exp[ii,] <- snr_exp[ii,] / snr_exp[1,]
}

snr_exp <- snr_exp %>%
  gather(key = "estimator", value = "score")

estimator2 <- c("chance", "upper", "diag", "SPoC", "Riemann") %>%
  factor(, levels = c("chance", "upper", "diag", "SPoC", "Riemann"))

snr_exp$estimator <- rep(estimator2, times = 10)

sigmas <- read.csv("./outputs/simuls/synth_snr/sigmas.csv", header = F)
snr_exp$xaxis <- rep(sigmas[["V1"]], each = 5)

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
  data = snr_exp %>% subset(estimator != "chance"),
  mapping = aes(y = score, x = xaxis, group = estimator,
                color = estimator)) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 4, shape = 21) +
  my_theme +
  scale_y_continuous(limits = c(0, 1.10), breaks = seq(0, 1, 0.25)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  scale_color_manual(values = color_cats, name = NULL) +
  # labs(x = TeX("distance between A and $I_p$"),
  #      y = "normalized M.A.E.") +
  geom_hline(yintercept = 1, color = "black", linetype = "dotted",
             size = 1) +
  annotate(geom = "text", x = 0.05, y = 1.04, label = "chance level",
           size = annotate_text_size) +
  labs(x = expression(sigma),
       y = "Normalized MAE") +
  theme(legend.position = "top", legend.title = element_text(size = 16))

fname <- "./outputs/fig_1b_snr_linear"
ggsave(paste0(fname, ".png"), width = 5, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"), width = 5, height = 5, dpi = 300,
  useDingbats = F)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))
