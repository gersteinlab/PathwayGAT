library(gplots)
library(pheatmap)

df_gene <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_gene.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_pathway_feature.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_microbe <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_microbe.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_gene <- read.table('../data/wikipath2021.tsv', header = F, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_microbe <- read.table('../data/wikipath_JHU_COAD_microbe.txt', header = F, sep = '\t', quote = '', stringsAsFactors = F)

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
#df_pathway_gene_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_gene_sum$pathway), width = 60, whitespace_only = F)
pathway_microbe_sum <- as.data.frame(rowSums(df_pathway[, 6970:ncol(df_pathway)]) / list_microbe_length, decreasing = T)
colnames(pathway_microbe_sum) <- c('value')
df_pathway_microbe_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_microbe_sum$value)
df_pathway_microbe_sum <- df_pathway_microbe_sum[order(df_pathway_microbe_sum$index, decreasing = T), ]
#df_pathway_microbe_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_microbe_sum$pathway), width = 60, whitespace_only = F)

gene_sum <- as.data.frame(sort(colSums(df_gene[, 2:ncol(df_gene)]), decreasing = T))
colnames(gene_sum) <- c('value')
df_gene_sum <- data.frame(gene = row.names(gene_sum), value = gene_sum$value, index = 1:nrow(gene_sum))
microbe_sum <- as.data.frame(sort(colSums(df_microbe[, 2:ncol(df_microbe)]), decreasing = T))
colnames(microbe_sum) <- c('value')
df_microbe_sum <- data.frame(microbe = row.names(microbe_sum), value = microbe_sum$value, index = 1:nrow(microbe_sum))

df_pathway_matrix <- as.matrix(df_pathway[, 2:ncol(df_pathway)])
colnames(df_pathway_matrix) <- colnames(df_pathway)[2:ncol(df_pathway)]
row.names(df_pathway_matrix) <- df_pathway$pathway

pathway_gene_list <- df_pathway_gene_sum$pathway[1:20]
gene_list <- df_gene_sum$gene[1:30]

gene_pathway_list <- c()
gene_in_pathway <- rep(FALSE, length(gene_list))

for (pathway in pathway_gene_list) {
    df_temp <- as.data.frame(df_pathway_matrix[pathway, 1:6968])
    colnames(df_temp) <- c('value')
    df_temp$gene <- row.names(df_temp)
    df_temp <- df_temp[which(df_temp$value > 0), ]
    temp_list <- row.names(df_temp)[order(df_temp$value, decreasing = T)][1:5]

    if (sum(temp_list %in% gene_list) > 0) {
        gene_in_pathway[which(gene_list %in% temp_list[temp_list %in% gene_list])] <- TRUE
        temp_list <- temp_list[-which(temp_list %in% gene_list)]
    }
    if (sum(temp_list %in% gene_pathway_list) > 0) {
        temp_list <- temp_list[-which(temp_list %in% gene_pathway_list)]
    }
    gene_pathway_list <- append(gene_pathway_list, temp_list)
}

write.table(c(gene_list, gene_pathway_list), 'JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_gene_list.txt', quote = F, row.names = F, col.names = F)

df_plot <- df_pathway_matrix[pathway_gene_list, c(gene_list, gene_pathway_list)]
add_string <- ifelse(gene_in_pathway, '**', '')
colnames(df_plot)[1:30] <- paste0(colnames(df_plot)[1:30], add_string)
add_string <- rep('*', length(gene_pathway_list))
colnames(df_plot)[31:ncol(df_plot)] <- paste0(colnames(df_plot)[31:ncol(df_plot)], add_string)

df_gene_list <- read.table('../result/diffexp_beta/JHU_COAD_final_total_gene_35_diffexp_beta.txt', sep = '\t', header = F, stringsAsFactors = F)
df_plot <- df_pathway_matrix[pathway_gene_list, df_gene_list$V1[1:50]]
df_plot <- df_plot[rowSums(df_plot) != 0, ]

df_gene_anno <- read.table('../data/NCG_cancerdrivers_annotation_supporting_evidence.tsv', sep = '\t', quote = '', header = T, stringsAsFactors = F)
COAD_gene_list <- unique(df_gene_anno$symbol[df_gene_anno$primary_site %in% c('colorectal', 'multiple')])
other_gene_list <- unique(df_gene_anno$symbol[-which(df_gene_anno$primary_site %in% c('colorectal', 'multiple'))])
other_gene_list <- other_gene_list[-which(other_gene_list %in% COAD_gene_list)]

annotation_col <- data.frame(cancer = ifelse(colnames(df_plot) %in% COAD_gene_list, 'COAD gene', ifelse(colnames(df_plot) %in% other_gene_list, 'other cancer gene', 'non-cancer gene')))
row.names(annotation_col) <- colnames(df_plot)
ann_colors = list(cancer = c('COAD gene' = 'tomato1', 'other cancer gene' = 'green1', 'non-cancer gene' = 'grey80'))

pheatmap(df_plot,
         show_rownames=T, show_colnames=T, cluster_cols=T, cluster_rows=F, cutree_cols=4,
         scale = "none", color = colorRampPalette(c("#ffffcc", "#fd8d3c", "#800026"))(1000), # Change color here
         border = FALSE, fontsize = 18, annotation_col = annotation_col, annotation_colors = ann_colors,
         filename = '../plot/JHU_final/gene_microbe_heatmap/JHU_COAD_final_35_top20_pathway_diff_beta_50_gene_color_cluster_NCG_multiple.pdf', height = 15, width = 35)

