library(ggplot2)

color_cats <- list(
  "black" = "#242424",
  "orange" = "#EFA435",
  "sky blue" = "#3EB6E7",
  "blueish green" = "#009D79",
  "yellow" = "#F2E55C",
  "blue" = "#0076B2",
  "vermillon" = "#E36C2F",
  "violet" = "#D683AB"
)

my_theme <- theme_minimal() + theme(
   text = element_text(family = 'Helvetica', size = 20),
   legend.text = element_text(size = 18),
   legend.title = element_text(size = 16),
   axis.text = element_text(size = 16)
)
theme_set(my_theme)

annotate_text_size <- 5
