library(ggplot2)

result <- read.table('files/TCGA_cancer_type_SNP_IPA/TCGA_cancer_type_SNP_associated_genes_IPA_pathway.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
result <- result[grepl('Cancer', result$Ingenuity.Canonical.Pathways) | grepl('cancer', result$Ingenuity.Canonical.Pathways), ]
result$Ingenuity.Canonical.Pathways <- as.factor(result$Ingenuity.Canonical.Pathways)
result <- result[order(result$X.log.p.value., decreasing = T), ]
ggplot(result) + geom_bar(stat="identity", width=0.6, aes(reorder(Ingenuity.Canonical.Pathways, X.log.p.value.),X.log.p.value.),fill="#2166ac",colour="#1d2a33") + 
  coord_flip() +
  labs(x="Ingenuity Canonical Pathways",y=expression(-log10(Pvalue)), title="IPA pathways") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(axis.title.x =element_text(size=24), 
        axis.title.y=element_text(size=24), 
        title=element_text(size=24)) +
  theme(axis.text.y = element_text(size = 20),
        axis.text.x = element_text(size = 20))

result <- read.table('files/JHU_COAD_final_gene_sum/JHU_COAD_diffexp_beta_IPA_pathway/JHU_COAD_diffexp_beta_35_IPA_disease_detail.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
result <- result[grepl('Colon ', result$Diseases.or.Functions.Annotation) | grepl('Colore', result$Diseases.or.Functions.Annotation) | grepl('colore', result$Diseases.or.Functions.Annotation) | grepl('Lung', result$Diseases.or.Functions.Annotation) | grepl('lung', result$Diseases.or.Functions.Annotation), ]
result$Diseases.or.Functions.Annotation <- as.factor(result$Diseases.or.Functions.Annotation)
ggplot(result) + geom_bar(stat="identity", width=0.6, aes(reorder(Diseases.or.Functions.Annotation, p.value, decreasing = T),-log10(p.value)),fill="#b2182b",colour="#1d2a33") + 
  coord_flip() +
  labs(x="Disease Annotation",y=expression(-log10(Pvalue)), title="IPA disease") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))  +
  theme(axis.title.x =element_text(size=24), 
        axis.title.y=element_text(size=24), 
        title=element_text(size=24)) +
  theme(axis.text.y = element_text(size = 20),
        axis.text.x = element_text(size = 20))