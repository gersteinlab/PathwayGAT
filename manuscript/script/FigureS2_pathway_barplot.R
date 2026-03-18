library(ggplot2)
library(RColorBrewer)

df_gene <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_gene.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_pathway_feature.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_SNP <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_SNP.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_gene <- read.table('../data/wikipath2021.tsv', header = F, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_SNP <- read.table('../data/wikipath_SNP.txt', header = F, sep = '\t', quote = '', stringsAsFactors = F)
num_pathway <- 20

list_gene_length <- c()
list_SNP_length <- c()

for (i in 1:nrow(df_pathway_gene)) {
    list_genes <- unlist(strsplit(df_pathway_gene[i, 2], ','))
    list_gene_length <- append(list_gene_length, length(list_genes))
    list_SNPs <- unlist(strsplit(df_pathway_SNP[i, 2], ','))
    list_SNP_length <- append(list_SNP_length, length(list_SNPs))
}

list_gene_length[list_gene_length == 0] <- .Machine$double.eps
pathway_gene_sum <- as.data.frame(rowSums(df_pathway[, 2:6990]) / list_gene_length, decreasing = T)
colnames(pathway_gene_sum) <- c('value')
df_pathway_gene_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_gene_sum$value)
df_pathway_gene_sum <- df_pathway_gene_sum[order(df_pathway_gene_sum$index, decreasing = T), ]
df_pathway_gene_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_gene_sum$pathway), width = 60, whitespace_only = F)
list_SNP_length[list_SNP_length == 0] <- .Machine$double.eps
pathway_SNP_sum <- as.data.frame(rowSums(df_pathway[, 6991:ncol(df_pathway)]) / list_SNP_length, decreasing = T)
colnames(pathway_SNP_sum) <- c('value')
df_pathway_SNP_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_SNP_sum$value)
df_pathway_SNP_sum <- df_pathway_SNP_sum[order(df_pathway_SNP_sum$index, decreasing = T), ]
df_pathway_SNP_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_SNP_sum$pathway), width = 60, whitespace_only = F)

a <- ggplot(df_pathway_gene_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_gene_sum$pathway)), y = index), stat = 'identity', 
             fill = '#f4a582') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    coord_flip()
ggsave('../plot/cancer_type_final/top_pathway_barplot/TCGA_cancer_type_SNP_top_pathway_gene_color.png', width = 2000, height = 1500, units = 'px')
a <- ggplot(df_pathway_gene_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_gene_sum$pathway)), y = index), stat = 'identity', 
             fill = '#f4a582') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    coord_flip()
ggsave('../plot/cancer_type_final/top_pathway_barplot/TCGA_cancer_type_SNP_top_pathway_gene_color.pdf', width = 3000, height = 2000, units = 'px')

a <- ggplot(df_pathway_SNP_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_SNP_sum$pathway)), y = index), stat = 'identity', 
             fill = '#91cf60') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    coord_flip()
ggsave('../plot/cancer_type_final/top_pathway_barplot/TCGA_cancer_type_SNP_top_pathway_SNP_color.png', width = 2000, height = 1500, units = 'px')
a <- ggplot(df_pathway_SNP_sum[1:num_pathway, ]) + 
    geom_bar(aes(x = factor(pathway, levels = rev(df_pathway_SNP_sum$pathway)), y = index), stat = 'identity', 
             fill = '#91cf60') + # Change color here
    xlab('Pathway') + 
    ylab('Explanation value') + 
    theme_classic() + 
    scale_y_continuous(expand = c(0, 0)) + 
    coord_flip()
ggsave('../plot/cancer_type_final/top_pathway_barplot/TCGA_cancer_type_SNP_top_pathway_SNP_color.pdf', width = 2000, height = 1500, units = 'px')
