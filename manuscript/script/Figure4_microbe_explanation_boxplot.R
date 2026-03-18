library(ggplot2)
library(RColorBrewer)

df_microbe_COAD <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_microbe.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_microbe_taxonomy <- read.table('../data/taxdump/rankedlineage.txt', sep = '\t', header = T, quote = '', comment.char = '', stringsAsFactors = F)

microbe_sum_COAD <- as.data.frame(sort(colSums(df_microbe_COAD[, 2:ncol(df_microbe_COAD)]), decreasing = T))
colnames(microbe_sum_COAD) <- c('value')
df_microbe_sum_COAD <- data.frame(microbe = row.names(microbe_sum_COAD), value = microbe_sum_COAD$value, index = 1:nrow(microbe_sum_COAD))

COAD_microbe_list <- c(562, 573, 28901, 287, 1280, 210, 197, 1590, 357276, 1351, 47715, 817, 1773, 40324, 485, 623, 727, 492670, 1747, 470, 1282)
df_microbe_taxonomy_COAD <- df_microbe_taxonomy[which(df_microbe_taxonomy$tax_id %in% COAD_microbe_list), ]
df_microbe_taxonomy_COAD$tax_name <- factor(df_microbe_taxonomy_COAD$tax_name, levels = df_microbe_taxonomy_COAD$tax_name)

df_microbe_class_COAD <- df_microbe_taxonomy_COAD[, c('tax_name', 'class')]
colnames(df_microbe_class_COAD) <- c('microbe', 'class')
df_microbe_class_COAD$microbe <- gsub(' ', '.', df_microbe_class_COAD$microbe)
df_microbe_sum_tax_COAD <- merge(df_microbe_sum_COAD, df_microbe_class_COAD, by = 'microbe', all.x = T)
df_microbe_sum_tax_COAD$class[is.na(df_microbe_sum_tax_COAD$class)] <- 'other'
df_microbe_sum_tax_COAD$class <- factor(df_microbe_sum_tax_COAD$class, levels = c('Actinomycetes', 'Bacteroidia', 'Bacilli', 'Betaproteobacteria', 'Epsilonproteobacteria', 'Gammaproteobacteria', 'other'))

a <- ggplot() + 
    geom_boxplot(data = df_microbe_sum_tax_COAD, mapping = aes(x = class, y = value)) + 
    geom_point(data = df_microbe_sum_tax_COAD[which(df_microbe_sum_tax_COAD$class != 'other'), ], mapping = aes(x = class, y = value, color = class), size = 3) + 
    scale_color_manual(values = c(brewer.pal(6,'Set1'), 'grey80')) + 
    scale_y_continuous(expand = c(0, 0)) +
    coord_cartesian(ylim = c(0, 25000)) + 
    xlab('microbe class') + 
    ylab('explanation value') + 
    theme_classic() + 
    theme(text = element_text(size = 15), axis.text.x = element_text(angle = 315, hjust = 0), legend.position = 'none')
ggsave('../plot/JHU_final/cancer_microbe_percent/JHU_COAD_final_35_microbe_class_relative_percent_boxplot.pdf', a, width = 9, height = 7.5)