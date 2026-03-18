library(ggplot2)
library(scales)

df_result <- read.table('gene_microbe_only/gene_microbe_only_result.txt', sep = ' ', quote = '', header = T, stringsAsFactors = F)
df_plot <- data.frame(model = rep(df_result$model, 3), ROC = c(df_result$ROC1, df_result$ROC2, df_result$ROC3))
df_plot$model <- as.factor(df_plot$model)
mean_ROC <- mean(as.numeric(df_result[1, c(3, 5, 7)]))
df_plot$ratio_ROC <- df_plot$ROC / mean_ROC

a <- ggplot(df_plot, aes(x = model, y = ratio_ROC, fill = model)) + 
    geom_boxplot() + 
    ylab('Relative AUC') + 
    scale_y_continuous(labels = percent_format(accuracy = 1)) + 
    theme_bw()
ggsave('../plot/JHU_final/gene_microbe_only/JHU_COAD_gene_microbe_only_relative_ROC.pdf', a, width = 6, height = 6, units = 'in')