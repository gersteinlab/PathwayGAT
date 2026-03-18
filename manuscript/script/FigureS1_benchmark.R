library(ggplot2)
library(scales)

df_randomforest <- read.table('benchmark/JHU_COAD_random_forest_result.txt', sep = ' ', quote = '', header = T, stringsAsFactors = F)
df_pathwaygat <- read.table('microbe_association/microbe_association_result.txt', sep = ' ', quote = '', header = T, stringsAsFactors = F)
df_XGBoost <- read.table('benchmark/JHU_COAD_XGBoost_result.txt', sep = '\t', quote = '', header = T, stringsAsFactors = F)
ROC_pathwaygat <- df_pathwaygat[which(df_pathwaygat$association == 0.15), c(4, 6, 8)]
ROC_randomforest <- df_randomforest[2, 5:7]
ROC_XGBoost <- df_XGBoost[1, 15:17]

df_plot <- data.frame(model = rep(c('PathwayGAT', 'Random Forest', 'XGBoost'), each = 3), ROC = c(as.numeric(ROC_pathwaygat), as.numeric(ROC_randomforest), as.numeric(ROC_XGBoost)))
df_plot$model <- as.factor(df_plot$model)
mean_ROC <- mean(as.numeric(ROC_pathwaygat))
df_plot$ratio_ROC <- df_plot$ROC / mean_ROC

a <- ggplot(df_plot, aes(x = model, y = ratio_ROC, fill = model)) + 
    geom_boxplot() + 
    ylab('Relative AUC') + 
    scale_y_continuous(labels = percent_format(accuracy = 1)) + 
    theme_bw()
ggsave('../plot/JHU_final/benchmark/JHU_COAD_benchmark_relative_ROC.pdf', a, width = 6, height = 6, units = 'in')