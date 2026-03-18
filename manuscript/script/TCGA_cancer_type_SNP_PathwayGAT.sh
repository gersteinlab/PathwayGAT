echo start at `date`

python main.py gene_SNP \
    --pathway_file ../data/wikipath2021.tsv \
    --gene_file ../data/TCGA_cancer_type_SNP_gene.txt \
    --SNP_file ../data/TCGA_cancer_type_SNP_SNP.txt \
    --SNP_coding ../data/TCGA_cancer_type_SNP_coding.filtered.txt \
    --SNP_noncoding ../data/TCGA_cancer_type_SNP_noncoding.filtered.txt \
    --meta_file ../data/TCGA_cancer_type_SNP_meta.txt \
    --class_name "investigation" \
    --output_prefix TCGA_cancer_type_SNP/TCGA_cancer_type_SNP \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_channels 64

python main.py evaluation \
    --dataset_file TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_training.pt \
    --meta_file ../data/TCGA_cancer_type_SNP_meta.txt \
    --class_name "investigation" \
    --output_prefix TCGA_cancer_type_SNP/TCGA_cancer_type_SNP \
    --learning_rate 0.001 \
    --hidden_channels 64

python main.py explanation \
    --dataset_file TCGA_cancer_type_SNP/TCGA_cancer_type_SNP_training.pt \
    --meta_file ../data/TCGA_cancer_type_SNP_meta.txt \
    --model_file TCGA_cancer_type_SNP/TCGA_cancer_type_SNP.final_model.pth \
    --class_name "investigation" \
    --output_prefix TCGA_cancer_type_SNP/TCGA_cancer_type_SNP \
    --hidden_channels 64

echo end at `date`