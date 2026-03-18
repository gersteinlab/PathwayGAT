library(tidyr)
library(gplots)
library(pheatmap)
library(RColorBrewer)

df_association <- read.table('../result/JHU_COAD_final_35_explanation_association_sample.sorted.txt', 
                             header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_gene_list <- read.table('../result/diffexp_beta/JHU_COAD_final_total_gene_35_diffexp_beta.txt', sep = '\t', header = F, stringsAsFactors = F)
gene_list <- df_gene_list$V1[1:50]
COAD_microbe_list <- c('Escherichia.coli', 'Klebsiella.pneumoniae', 'Salmonella.enterica', 'Pseudomonas.aeruginosa', 'Staphylococcus.aureus', 'Helicobacter.pylori', 'Campylobacter.jejuni', 'Lactiplantibacillus.plantarum', 'Phocaeicola.dorei', 'Enterococcus.faecalis', 'Lacticaseibacillus.rhamnosus', 'Bacteroides.fragilis', 'Mycobacterium.tuberculosis')
other_microbe_list <- c('Stenotrophomonas.maltophilia', 'Neisseria.gonorrhoeae', 'Shigella.flexneri', 'Haemophilus.influenzae', 'Bacillus.velezensis', 'Cutibacterium.acnes', 'Acinetobacter.baumannii', 'Mycobacterium.avium', 'Enterobacter.hormaechei')
microbe_list <- c(COAD_microbe_list, other_microbe_list)
df_association_select <- df_association[which(df_association$microbe %in% microbe_list), ]
df_association_select <- df_association_select[which(df_association_select$gene %in% gene_list), c('gene', 'microbe', 'pearson')]
df_association_select <- df_association_select %>% pivot_wider(names_from = microbe, values_from = pearson)
df_association_matrix <- as.matrix(df_association_select[, 2:ncol(df_association_select)])
row.names(df_association_matrix) <- df_association_select$gene
df_association_matrix <- t(df_association_matrix)

pheatmap(df_association_matrix,
         show_rownames=T, show_colnames=T, cluster_cols=T, cluster_rows=T, 
         scale = "none", color = colorRampPalette(c("#2166ac", "#f7f7f7", "#b2182b"))(1000), # Change color here
         border = FALSE, fontsize = 20,
         filename = '../plot/JHU_final/gene_microbe_explanation_association_heatmap/JHU_COAD_final_35_cancer_gene_cancer_microbe_sample_association_none.pdf', height = 8, width = 16)