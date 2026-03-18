library(ggplot2)
library(igraph)
library(ggraph)
library(ggrepel)
library(dplyr)

df_attention <- read.table('../result/TCGA_cancer_type_SNP_attention1.txt', sep = '\t', quote = '', header = T, stringsAsFactors = F)
df_pathway_gene <- read.table('../data/wikipath2021.tsv', header = F, sep = '\t', quote = '', stringsAsFactors = F)

list_nodes <- c(df_top_attention_name$source, df_top_attention_name$destination)
df_top_attention_node <- as.data.frame(table(list_nodes))

df_gene <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_gene.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_pathway_feature.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_SNP <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_SNP.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_gene <- read.table('../data/wikipath2021.tsv', header = F, sep = '\t', quote = '', stringsAsFactors = F)
df_pathway_SNP <- read.table('../data/wikipath_SNP.txt', header = F, sep = '\t', quote = '', stringsAsFactors = F)

list_gene_length <- c()
list_SNP_length <- c()

for (i in 1:nrow(df_pathway_gene)) {
    list_genes <- unlist(strsplit(df_pathway_gene[i, 2], ','))
    list_gene_length <- append(list_gene_length, length(list_genes))
    list_SNPs <- unlist(strsplit(df_pathway_SNP[i, 2], ','))
    list_SNP_length <- append(list_SNP_length, length(list_SNPs))
}

pathway_gene_sum <- as.data.frame(rowSums(df_pathway[, 2:6990]) / list_gene_length, decreasing = T)
colnames(pathway_gene_sum) <- c('value')
df_pathway_gene_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_gene_sum$value)
df_pathway_gene_sum <- df_pathway_gene_sum[order(df_pathway_gene_sum$index, decreasing = T), ]
df_pathway_gene_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_gene_sum$pathway), width = 60, whitespace_only = F)
pathway_SNP_sum <- as.data.frame(rowSums(df_pathway[, 6991:ncol(df_pathway)]) / list_SNP_length, decreasing = T)
colnames(pathway_SNP_sum) <- c('value')
df_pathway_SNP_sum <- data.frame(pathway = df_pathway$pathway, index = pathway_SNP_sum$value)
df_pathway_SNP_sum <- df_pathway_SNP_sum[order(df_pathway_SNP_sum$index, decreasing = T), ]
df_pathway_SNP_sum$pathway <- stringr::str_wrap(gsub('\\.', ' ', df_pathway_SNP_sum$pathway), width = 60, whitespace_only = F)

gene_sum <- as.data.frame(sort(colSums(df_gene[, 2:ncol(df_gene)]), decreasing = T))
colnames(gene_sum) <- c('value')
df_gene_sum <- data.frame(gene = row.names(gene_sum), value = gene_sum$value, index = 1:nrow(gene_sum))
SNP_sum <- as.data.frame(sort(colSums(df_SNP[, 2:ncol(df_SNP)]), decreasing = T))
colnames(SNP_sum) <- c('value')
df_SNP_sum <- data.frame(SNP = row.names(SNP_sum), value = SNP_sum$value, index = 1:nrow(SNP_sum))

df_pathway_matrix <- as.matrix(df_pathway[, 2:ncol(df_pathway)])
colnames(df_pathway_matrix) <- colnames(df_pathway)[2:ncol(df_pathway)]
row.names(df_pathway_matrix) <- stringr::str_wrap(gsub('\\.', ' ', df_pathway$pathway), width = 60, whitespace_only = F)

pathway_gene_list <- df_pathway_gene_sum$pathway[1:20]
gene_list <- df_gene_sum$gene[1:30]

df_attention_name <- df_attention
for (i in 1:nrow(df_attention)) {
    df_attention_name[i, 1] <- df_pathway_gene$V1[df_attention[i, 1] + 1]
    df_attention_name[i, 2] <- df_pathway_gene$V1[df_attention[i, 2] + 1]
}

df_attention_name_filtered <- df_attention_name[which(df_attention_name$source %in% pathway_gene_list & df_attention_name$destination %in% pathway_gene_list), ]
df_attention_name_filtered$source <- stringr::str_wrap(gsub('\\.', ' ', df_attention_name_filtered$source), width = 30, whitespace_only = T)
df_attention_name_filtered$destination <- stringr::str_wrap(gsub('\\.', ' ', df_attention_name_filtered$destination), width = 30, whitespace_only = T)

options(ggrepel.max.overlaps = Inf)
non_self_nodes <- unique(c(df_attention_name_filtered$source[df_attention_name_filtered$source != df_attention_name_filtered$destination],
                           df_attention_name_filtered$destination[df_attention_name_filtered$source != df_attention_name_filtered$destination]))
df_node_size <- read.table('../result/cancer_type_top_gene_pathway_enrichment.txt', sep = '\t', quote = '', header = F, stringsAsFactors = F)
colnames(df_node_size) <- c('node_name', 'node_pvalue')
df_node_size$node_size <- -1 * log10(df_node_size$node_pvalue)
df_node_size$node_name <- stringr::str_wrap(gsub('\\.', ' ', df_node_size$node_name), width = 30, whitespace_only = T)

g <- graph_from_data_frame(df_attention_name_filtered, directed = TRUE, vertices = df_node_size)
ggraph(g, layout = "kk") +
  geom_edge_loop(aes(width = attention, color = attention), alpha = 0.8, show.legend = TRUE) + 
  geom_edge_link(aes(width = attention, color = attention), alpha = 0.8, show.legend = TRUE) +
  geom_node_point(aes(size = node_size), color = "steelblue") +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  scale_edge_width(range = c(0.5, 3)) +
  scale_edge_color_gradient(low = "lightgray", high = "red") +
  scale_size_continuous(name = "-log10(p)", range = c(1, 7)) +
  theme_void()

df_attention_name_filtered <- df_attention_name_filtered %>%
  filter(source %in% non_self_nodes | destination %in% non_self_nodes)
df_node_size <- df_node_size %>% filter(node_name %in% non_self_nodes)

g <- graph_from_data_frame(df_attention_name_filtered, directed = TRUE, vertices = df_node_size)
a <- ggraph(g, layout = "kk") +
  geom_edge_link(aes(width = attention, color = attention), alpha = 0.8, show.legend = TRUE) +
  geom_node_point(aes(size = node_size), color = "steelblue") +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  scale_edge_width(range = c(0.5, 3)) +
  scale_edge_color_gradient(low = "lightgray", high = "red") +
  scale_size_continuous(name = "-log10(p)", range = c(1, 7)) +
  theme_void()
ggsave('../plot/JHU_final/top_pathway_attention/TCGA_cancer_type_final_attention1_top_gene_size_noloop.pdf', a, width = 10, height = 10, units = 'in')