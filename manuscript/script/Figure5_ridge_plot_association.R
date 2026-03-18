library(ggplot2)
library(ggridges)

df_SNP_list <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_top_SNP_total.txt', sep = '\t', quote = '', header = F, stringsAsFactors = F)
df_gene_list <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_top_SNP_total_gene.txt', sep = '\t', quote = '', header = F, stringsAsFactors = F)
df_noncoding <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_noncoding.score.txt', header = F, sep = '\t', quote = '', stringsAsFactors = F)
df_coding <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_coding.score.txt', header = F, sep = '\t', quote = '', stringsAsFactors = F)

df_coding_filtered <- df_coding[which(df_coding$V1 %in% df_SNP_list$V1), c('V2', 'V3')]
df_plot <- df_coding_filtered
df_plot$V2 <- factor(df_plot$V2, levels = unique(df_plot$V2))
df_plot$V3 <- as.numeric(df_plot$V3)

df_gene_anno <- read.table('../data/NCG_cancerdrivers_annotation_supporting_evidence.tsv', sep = '\t', quote = '', header = T, stringsAsFactors = F)
cancer_gene_list <- unique(df_gene_anno$symbol)

a <- ggplot(df_plot, aes(x = V3, y = V2)) +
  geom_density_ridges2(stat = "binline", binwidth = 0.005, scale = 0.8) + 
  geom_density_ridges2(scale = 0.4, bandwidth = 0.1, alpha = 0.5) + 
  scale_x_continuous(
    breaks = c(0, 1, 2, 3, 4), limits = c(-.5, 5),
    expand = c(0, 0), name = "FunSeq score"
  ) + 
  scale_y_discrete(expand = expansion(add = c(0, 1.)), name = "Genes & score density", labels = paste0(unique(df_plot$V2), ifelse(unique(df_plot$V2) %in% cancer_gene_list, '*', ''))) + 
  theme_ridges(grid = FALSE) + 
  theme(
    axis.title.x = element_text(hjust = 0.5, size = 20),
    axis.title.y = element_text(hjust = 0.5, size = 20),
    axis.text.y = element_text(size = 20),
    legend.position = 'none'
)
ggsave('../plot/cancer_type_final/ridge_plot_density/TCGA_cancer_type_SNP_SNP_coding_associated_gene_ridge_plot_density_size.png', a, width = 2000, height = 1000, units = 'px')
ggsave('../plot/cancer_type_final/ridge_plot_density/TCGA_cancer_type_SNP_SNP_coding_associated_gene_ridge_plot_density_size.pdf', a, width = 10, height = 5)

df_noncoding_filtered <- df_noncoding[which(df_noncoding$V1 %in% df_SNP_list$V1), c('V2', 'V3')]
df_plot <- df_noncoding_filtered
df_plot$V2 <- factor(df_plot$V2, levels = unique(df_plot$V2))
df_plot$V3 <- as.numeric(df_plot$V3)

a <- ggplot(df_plot, aes(x = V3, y = V2)) +
  geom_density_ridges2(stat = "binline", binwidth = 0.005, scale = 0.8) + 
  geom_density_ridges2(scale = 0.4, bandwidth = 0.1, alpha = 0.5) + 
  scale_x_continuous(
    breaks = c(0, 1, 2, 3, 4), limits = c(-.5, 5),
    expand = c(0, 0), name = "FunSeq score"
  ) + 
  scale_y_discrete(expand = expansion(add = c(0, 1.)), name = "Genes & score density", labels = paste0(unique(df_plot$V2), ifelse(unique(df_plot$V2) %in% cancer_gene_list, '*', ''))) + 
  theme_ridges(grid = FALSE) + 
  theme(
    axis.title.x = element_text(hjust = 0.5, size = 20),
    axis.title.y = element_text(hjust = 0.5, size = 20),
    axis.text.y = element_text(size = 20),
    legend.position = 'none'
)
ggsave('../plot/cancer_type_final/ridge_plot_density/TCGA_cancer_type_SNP_SNP_noncoding_associated_gene_ridge_plot_density_size.png', a, width = 2000, height = 2000, units = 'px')
ggsave('../plot/cancer_type_final/ridge_plot_density/TCGA_cancer_type_SNP_SNP_noncoding_associated_gene_ridge_plot_density_size.pdf', a, width = 10, height = 10)