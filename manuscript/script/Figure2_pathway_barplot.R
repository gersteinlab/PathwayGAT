library(ggplot2)
library(RColorBrewer)

df_gene <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_gene.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_pathway_feature.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_microbe <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_microbe.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_gene <- read.table('../data/wikipath2021.tsv', header = F, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_microbe <- read.table('../data/wikipath_JHU_COAD_microbe.txt', header = F, sep = '\t', quote = '', stringsAsFactors = F)
num_pathway <- 20

list_gene_length <- c()
list_microbe_length <- c()

for (i in 1:nrow(df_pathway_gene)) {
    list_genes <- unlist(strsplit(df_pathway_gene[i, 2], ','))
    list_gene_length <- append(list_gene_length, length(list_genes))
    list_microbes <- unlist(strsplit(df_pathway_microbe[i, 2], ','))
    list_microbe_length <- append(list_microbe_length, length(list_microbes))
}

pathway_gene_sum <- as.data.frame(rowSums(df_pathway[, 2:6969]) / list_gene_length, decreasing = T)
colnames(pathway_gene_sum) <- c('value')
df_pathway_gene_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_gene_sum$value)
df_pathway_gene_sum <- df_pathway_gene_sum[order(df_pathway_gene_sum$index, decreasing = T), ]
df_pathway_gene_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_gene_sum$pathway), width = 50, whitespace_only = F)
pathway_microbe_sum <- as.data.frame(rowSums(df_pathway[, 6970:ncol(df_pathway)]) / list_microbe_length, decreasing = T)
colnames(pathway_microbe_sum) <- c('value')
df_pathway_microbe_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_microbe_sum$value)
df_pathway_microbe_sum <- df_pathway_microbe_sum[order(df_pathway_microbe_sum$index, decreasing = T), ]
df_pathway_microbe_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_microbe_sum$pathway), width = 50, whitespace_only = F)

a <- ggplot(df_pathway_gene_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_gene_sum$pathway)), y = index), stat = 'identity', 
             fill = '#f4a582') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    theme(text = element_text(size = 20)) +
    scale_y_continuous(expand = c(0, 0)) + 
    coord_flip()
ggsave('../plot/JHU_final/top_pathway_barplot/JHU_COAD_final_35_top_pathway_gene_color.png', width = 2000, height = 1500, units = 'px')
a <- ggplot(df_pathway_gene_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_gene_sum$pathway)), y = index), stat = 'identity', 
             fill = '#f4a582') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    theme(text = element_text(size = 20)) +
    coord_flip()
ggsave('../plot/JHU_final/top_pathway_barplot/JHU_COAD_final_35_top_pathway_gene_color.pdf', width = 3000, height = 3000, units = 'px')

a <- ggplot(df_pathway_microbe_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_microbe_sum$pathway)), y = index), stat = 'identity', 
             fill = '#92c5de') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    theme(text = element_text(size = 20)) +
    coord_flip()
ggsave('../plot/JHU_final/top_pathway_barplot/JHU_COAD_final_35_top_pathway_microbe_color.png', width = 2000, height = 1500, units = 'px')
a <- ggplot(df_pathway_microbe_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_microbe_sum$pathway)), y = index), stat = 'identity', 
             fill = '#92c5de') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    theme(text = element_text(size = 20)) +
    coord_flip()
ggsave('../plot/JHU_final/top_pathway_barplot/JHU_COAD_final_35_top_pathway_microbe_color.pdf', width = 3000, height = 3000, units = 'px')
