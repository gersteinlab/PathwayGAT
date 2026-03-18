echo start at `date`

python main.py microbe_gene \
    --pathway_file ../data/wikipath2021.tsv \
    --microbe_file ../data/TCGA_COAD.microbe.txt \
    --microbe_gene_file ../data/TCGA_COAD.gene_microbe_association.txt \
    --gene_file ../data/TCGA_COAD.gene.txt \
    --meta_file ../data/COAD_sample_metadata.processed.txt \
    --class_name "sample_type2" \
    --output_prefix JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35 \
    --microbe_corr_threshold 0.15 \
    --batch_size 32 \
    --hidden_channels 128 \
    --learning_rate 0.001 \
    --epochs 35

python main.py evaluation \
    --dataset_file JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_training.pt \
    --meta_file ../data/COAD_sample_metadata.processed.txt \
    --class_name "sample_type2" \
    --output_prefix JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35 \
    --batch_size 32 \
    --hidden_channels 128 \
    --learning_rate 0.001 \
    --epochs 35

python main.py explanation \
    --dataset_file JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_training.pt \
    --meta_file ../data/COAD_sample_metadata.processed.txt \
    --model_file JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35.final_model.pth \
    --class_name "sample_type2" \
    --output_prefix JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35 \
    --hidden_channels 128

echo end at `date`