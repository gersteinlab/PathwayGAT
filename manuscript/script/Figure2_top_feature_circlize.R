library(circlize)
library(RColorBrewer)

df_gene <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_gene.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)
df_microbe <- read.table('JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_explanation_microbe.txt', header = T, sep = '\t', quote = '', stringsAsFactors = F)

gene_sum <- as.data.frame(sort(colSums(df_gene[, 2:ncol(df_gene)]), decreasing = T))
colnames(gene_sum) <- c('value')
df_gene_sum <- data.frame(gene = row.names(gene_sum), value = gene_sum$value)
microbe_sum <- as.data.frame(sort(colSums(df_microbe[, 2:ncol(df_microbe)]), decreasing = T))
colnames(microbe_sum) <- c('value')
df_microbe_sum <- data.frame(microbe = row.names(microbe_sum), value = microbe_sum$value)
df_microbe_sum$microbe <- stringr::str_wrap(gsub('\\.', ' ', df_microbe_sum$microbe), width = 30, whitespace_only = F)

df_gene_anno <- read.table('../data/NCG_cancerdrivers_annotation_supporting_evidence.tsv', sep = '\t', quote = '', header = T, stringsAsFactors = F)
COAD_gene_list <- unique(df_gene_anno$symbol[df_gene_anno$primary_site %in% c('colorectal', 'multiple')])

pdf('../plot/JHU_final/top_gene_microbe_circlize/JHU_COAD_final_35_top50_gene_circlize.pdf', width = 10, height = 10)

circos.clear()
circos.par("gap.degree" = 20, "cell.padding" = c(0, 0, 0, 0), "start.degree" = 90, "circle.margin" = c(0.6))
circos.par(points.overflow.warning = FALSE)
circos.initialize(factors = 1, xlim = c(0, 1), sector.width = 1)
circos.trackPlotRegion(factors = 1, ylim = c(1000, 6000),
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
                   y = 6100,
                   labels = df_gene_sum$gene[j],
                   1,
                   col = 'black',
                   facing = "clockwise", niceFacing = TRUE, adj = c(0, 0),
                   cex = 2.4)
}

dev.off()

pdf('../plot/JHU_final/top_gene_microbe_circlize/JHU_COAD_final_35_top50_microbe_circlize.pdf', width = 10, height = 10)

circos.clear()
circos.par("gap.degree" = 20, "cell.padding" = c(0, 0, 0, 0), "start.degree" = 90, "circle.margin" = c(0.9))
circos.par(points.overflow.warning = FALSE)
circos.initialize(factors = 1, xlim = c(0, 1), sector.width = 1)
circos.trackPlotRegion(factors = 1, ylim = c(10000, 23000),
                       track.height = 0.5, bg.border = "grey",
                       bg.col = "grey96")
circos.yaxis('left')

for (j in 1:50) {
  circos.trackPoints(factors = 1,
                     x = seq(0, 1, length.out = 50 + 1)[j],
                     y = df_microbe_sum$value[j],
                     1,
                     col = 'grey40',
                     pch = 19)
  circos.trackLines(factors = 1,
                    x = seq(0, 1, length.out = 50 + 1)[j],
                    y = df_microbe_sum$value[j],
                    1,
                    col = 'grey40',
                    type = "h")
  circos.trackText(factors = 1,
                   x = seq(0, 1, length.out = 50 + 1)[j],
                   y = 24000,
                   labels = df_microbe_sum$microbe[j],
                   1,
                   col = 'black',
                   facing = "clockwise", niceFacing = TRUE, adj = c(0, 0),
                   cex = 2)
}

dev.off()