echo test_PathwayGAT start at `date`

mkdir -p ../output

python main.py microbe_gene_SNP \
    --pathway_file pathway_info.txt \
    --microbe_file microbe_profile.txt \
    --gene_file gene_profile.txt \
    --SNP_file SNP_profile.txt \
    --SNP_coding SNP_coding_example.txt \
    --SNP_noncoding SNP_noncoding_example.txt \
    --microbe_gene_file microbe_gene_association.txt \
    --meta_file meta_info.txt \
    --class_name "investigation" \
    --output_prefix ../output/test_PathwayGAT \
    --microbe_corr_threshold 0.1 \
    --batch_size 8 \
    --hidden_channels 16 \
    --learning_rate 0.0001 \
    --epochs 5

python main.py evaluation \
    --dataset_file ../output/test_PathwayGAT_training.pt \
    --meta_file meta_info.txt \
    --class_name "investigation" \
    --output_prefix ../output/test_PathwayGAT \
    --batch_size 8 \
    --hidden_channels 16 \
    --learning_rate 0.0001 \
    --epochs 5

python main.py explanation \
    --node_file ../output/test_PathwayGAT_nodes.pt \
    --meta_file meta_info.txt \
    --model_file ../output/test_PathwayGAT.best_model.pth \
    --pathway_file pathway_info.txt \
    --class_name "investigation" \
    --output_prefix ../output/test_PathwayGAT \
    --hidden_channels 16

echo test_PathwayGAT end at `date`
