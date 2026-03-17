echo start at `date`

module load miniconda
conda activate test_pathwaygat

pathwaygat modules \
    --pathway_file pathway_info.txt \
    --gene_file gene_profile.txt \
    --module_file_list example_modules_list.txt \
    --module_gene_file_list example_modules_gene_list.txt \
    --meta_file meta_info.txt \
    --class_name "sample_type2" \
    --output_prefix example_modules_output \
    --corr_threshold_list example_modules_corr_threshold_list.txt \
    --batch_size 32 \
    --hidden_channels 128 \
    --learning_rate 0.001 \
    --epochs 30
  
echo end at `date`
