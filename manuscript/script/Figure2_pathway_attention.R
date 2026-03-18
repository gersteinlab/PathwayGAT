library(ggplot2)
library(igraph)
library(ggraph)
library(ggrepel)
library(dplyr)

df_attention <- read.table('attention/JHU_COAD_final_35_attention1.txt', sep = '\t', quote = '', header = T, stringsAsFactors = F)
df_top_attention <- head(df_attention[order(df_attention$attention, decreasing = T), ], 100)
df_pathway_gene <- read.table('../data/wikipath2021.tsv', header = F, sep = '\t', quote = '', stringsAsFactors = F)

df_top_attention_name <- df_top_attention
for (i in 1:nrow(df_top_attention)) {
    df_top_attention_name[i, 1] <- df_pathway_gene$V1[df_top_attention[i, 1] + 1]
    df_top_attention_name[i, 2] <- df_pathway_gene$V1[df_top_attention[i, 2] + 1]
}
list_nodes <- c(df_top_attention_name$source, df_top_attention_name$destination)
df_top_attention_node <- as.data.frame(table(list_nodes))

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
df_attention_name <- df_attention
for (i in 1:nrow(df_attention)) {
    df_attention_name[i, 1] <- df_pathway_gene$V1[df_attention[i, 1] + 1]
    df_attention_name[i, 2] <- df_pathway_gene$V1[df_attention[i, 2] + 1]
}
df_attention_name_filtered <- df_attention_name[which(df_attention_name$source %in% pathway_gene_list & df_attention_name$destination %in% pathway_gene_list), ]

df_attention_name_filtered$source <- stringr::str_wrap(gsub('\\.', ' ', df_attention_name_filtered$source), width = 30, whitespace_only = T)
df_attention_name_filtered$destination <- stringr::str_wrap(gsub('\\.', ' ', df_attention_name_filtered$destination), width = 30, whitespace_only = T)

df_node_size <- read.table('../result/JHU_COAD_top_gene_pathway_enrichment.txt', sep = '\t', quote = '', header = F, stringsAsFactors = F)
colnames(df_node_size) <- c('node_name', 'node_pvalue')
df_node_size$node_size <- -1 * log10(df_node_size$node_pvalue)
df_node_size$node_name <- stringr::str_wrap(gsub('\\.', ' ', df_node_size$node_name), width = 30, whitespace_only = T)

g <- graph_from_data_frame(df_attention_name_filtered, directed = TRUE, vertices = df_node_size)
a <- ggraph(g, layout = "kk") +
  geom_edge_loop(aes(width = attention, color = attention), alpha = 0.8, show.legend = TRUE) + 
  geom_edge_link(aes(width = attention, color = attention), alpha = 0.8, show.legend = TRUE) +
  geom_node_point(aes(size = node_size), color = "steelblue") +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  scale_edge_width(range = c(0.5, 3)) +
  scale_edge_color_gradient(low = "lightgray", high = "red") +
  scale_size_continuous(name = "-log10(p)", range = c(1, 7)) +
  theme_void()
ggsave('../plot/JHU_final/top_pathway_attention/JHU_COAD_final_attention1_top_gene_size_loop.pdf', a, width = 10, height = 10, units = 'in')

non_self_nodes <- unique(c(df_attention_name_filtered$source[df_attention_name_filtered$source != df_attention_name_filtered$destination],
                           df_attention_name_filtered$destination[df_attention_name_filtered$source != df_attention_name_filtered$destination]))

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
ggsave('../plot/JHU_final/top_pathway_attention/JHU_COAD_final_attention1_top_gene_size_noloop.pdf', a, width = 10, height = 10, units = 'in')