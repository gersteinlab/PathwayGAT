library(circlize)
library(RColorBrewer)

df_gene <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_gene.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_SNP <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_explanation_SNP.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)

gene_sum <- as.data.frame(sort(colSums(df_gene[, 2:ncol(df_gene)]), decreasing = T))
colnames(gene_sum) <- c('value')
df_gene_sum <- data.frame(gene = row.names(gene_sum), value = gene_sum$value)
SNP_sum <- as.data.frame(sort(colSums(df_SNP[, 2:ncol(df_SNP)]), decreasing = T))
colnames(SNP_sum) <- c('value')
df_SNP_sum <- data.frame(SNP = row.names(SNP_sum), value = SNP_sum$value)

df_coding <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_coding.filtered.txt', sep = '\t', quote = '', header = F, stringsAsFactors = F)
df_noncoding <- read.table('TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_noncoding.filtered.txt', sep = '\t', quote = '', header = F, stringsAsFactors = F)
df_coding_sum <- df_SNP_sum[which(df_SNP_sum$SNP %in% df_coding$V1), ]
df_noncoding_sum <- df_SNP_sum[which(df_SNP_sum$SNP %in% df_noncoding$V1), ]
df_SNP_sum_coding <- rbind(df_coding_sum[1:50, ], df_noncoding_sum[which(df_noncoding_sum$value > 0), ])
df_SNP_sum_coding$SNP[51:nrow(df_SNP_sum_coding)] <- paste0('*', df_SNP_sum_coding$SNP[51:nrow(df_SNP_sum_coding)])

pdf('../plot/cancer_type_final/top_feature_circlize/TCGA_cancer_type_SNP_top50_gene_circlize.pdf', width = 10, height = 10)

circos.clear()
circos.par("gap.degree" = 20, "cell.padding" = c(0, 0, 0, 0), "start.degree" = 90, "circle.margin" = c(0.6))
circos.par(points.overflow.warning = FALSE)
circos.initialize(factors = 1, xlim = c(0, 1), sector.width = 1)
circos.trackPlotRegion(factors = 1, ylim = c(6000, 22000),
                       track.height = 0.5, bg.border = "grey",
                       bg.col = "grey96")
circos.yaxis('left')

for (j in 1:50) {
  circos.trackPoints(factors = 1,
                     x = seq(0, 1, length.out = 50 + 1)[j],
                     y = df_gene_sum$value[j],
                     1,
                     col = 'grey40',
                     pch = 19)
  circos.trackLines(factors = 1,
                    x = seq(0, 1, length.out = 50 + 1)[j],
                    y = df_gene_sum$value[j],
                    1,
                    col = 'grey40',
                    type = "h")
  circos.trackText(factors = 1,
                   x = seq(0, 1, length.out = 50 + 1)[j],
                   y = 23000,
                   labels = df_gene_sum$gene[j],
                   1,
                   col = 'black',
                   facing = "clockwise", niceFacing = TRUE, adj = c(0, 0),
                   cex = 2)
}

dev.off()

pdf('../plot/cancer_type_final/top_feature_circlize/TCGA_cancer_type_SNP_top_SNP_coding_circlize.pdf', width = 10, height = 10)

circos.clear()
circos.par("gap.degree" = 20, "cell.padding" = c(0, 0, 0, 0), "start.degree" = 90, "circle.margin" = c(0.6))
circos.par(points.overflow.warning = FALSE)
circos.initialize(factors = 1, xlim = c(0, 1), sector.width = 1)
circos.trackPlotRegion(factors = 1, ylim = c(0, 600),
                       track.height = 0.5, bg.border = "grey",
                       bg.col = "grey96")
circos.yaxis('left')

for (j in 1:nrow(df_SNP_sum_coding)) {
  circos.trackPoints(factors = 1,
                     x = seq(0, 1, length.out = nrow(df_SNP_sum_coding) + 1)[j],
                     y = df_SNP_sum_coding$value[j],
                     1,
                     col = 'grey40',
                     pch = 19)
  circos.trackLines(factors = 1,
                    x = seq(0, 1, length.out = nrow(df_SNP_sum_coding) + 1)[j],
                    y = df_SNP_sum_coding$value[j],
                    1,
                    col = 'grey40',
                    type = "h")
  circos.trackText(factors = 1,
                   x = seq(0, 1, length.out = nrow(df_SNP_sum_coding) + 1)[j],
                   y = 650,
                   labels = df_SNP_sum_coding$SNP[j],
                   1,
                   col = 'black',
                   facing = "clockwise", niceFacing = TRUE, adj = c(0, 0),
                   cex = 1)
}

dev.off()
