library(ggplot2)
library(ggsignif)
library(scales)

df_result <- read.table('../../pathwaygat/microbe_association/microbe_association_result.txt', sep = ' ', quote = '', header = T, stringsAsFactors = F)
df_plot_ROC <- data.frame(association = rep(df_result$association, 3), ROC = c(df_result$ROC1, df_result$ROC2, df_result$ROC3))
df_plot_ROC$association <- as.factor(df_plot_ROC$association)
mean_ROC <- mean(df_plot_ROC[which(df_plot_ROC$association == 0.15), 2])
df_plot_ROC$ratio <- df_plot_ROC$ROC / mean_ROC
a <- ggplot(df_plot_ROC, aes(x = association, y = ratio, fill = association)) + 
    geom_boxplot() + 
    ylab('Relative AUC') + 
    scale_y_continuous(labels = percent_format(accuracy = 1)) + 
    theme_bw()
ggsave('../plot/JHU_final/microbe_association/JHU_COAD_microbe_association_relative_ROC.pdf', a, width = 6, height = 6, units = 'in')