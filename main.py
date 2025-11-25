import argparse
from training import *
from evaluation import *
from explanation import *

# Created by Weihao to allow read parameters from command line

def main():
    parser = argparse.ArgumentParser(description="PathwayGAT is a graph neural network model that can predict biological phenotypes based on constructed biological networks consisted of pathways and related gene, microbe, and SNPs.")
    subparsers = parser.add_subparsers(title='modules', help='Different modules of this program', required=True, dest='subparser_name')
    
    parser_gene = subparsers.add_parser('gene', help='Train the model with gene data')
    parser_gene.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_gene.add_argument('--gene_file', help='Specify the file that contains gene expression matrix.')
    parser_gene.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_gene.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_gene.add_argument('--output_prefix', help='Specify the output prefix of the saved model parameters and dataset.')
    parser_gene.add_argument('--batch_size', default=32, help='Specify the batch size during training. Default set to 32.')
    parser_gene.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during training. Default set to 32.')
    parser_gene.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during training. Default set to 0.005.')
    parser_gene.add_argument('--epochs', default=20, help='Specify the epochs during training. Default set to 20.')
    parser_gene.add_argument('--sample_number', default=0, help='Specify the number of samples you want to use during training. Useful for downsampling. Default set to 0 (no downsampling).')
    parser_gene.add_argument('--seed', default=42, help='Specify the seed you want to use during training. Default set to 42.')

    parser_microbe = subparsers.add_parser('microbe', help='Train the model with microbe data')
    parser_microbe.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_microbe.add_argument('--microbe_file', help='Specify the file that contains microbe matrix.')
    parser_microbe.add_argument('--microbe_gene_file', help='Specify the file that contains microbe gene association.')
    parser_microbe.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_microbe.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_microbe.add_argument('--output_prefix', help='Specify the output prefix of the saved model parameters and dataset.')
    parser_microbe.add_argument('--microbe_corr_threshold', default=0.05, help='Specify the correlation threshold for microbe to be considered assocated to a specific gene. Default set to 0.05')
    parser_microbe.add_argument('--batch_size', default=32, help='Specify the batch size during training. Default set to 32.')
    parser_microbe.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during training. Default set to 32.')
    parser_microbe.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during training. Default set to 0.005.')
    parser_microbe.add_argument('--epochs', default=20, help='Specify the epochs during training. Default set to 20.')
    parser_microbe.add_argument('--sample_number', default=0, help='Specify the number of samples you want to use during training. Useful for downsampling. Default set to 0 (no downsampling).')
    parser_microbe.add_argument('--seed', default=42, help='Specify the seed you want to use during training. Default set to 42.')

    parser_microbe_gene = subparsers.add_parser('microbe_gene', help='Train the model with microbe and gene expression data')
    parser_microbe_gene.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_microbe_gene.add_argument('--microbe_file', help='Specify the file that contains microbe matrix.')
    parser_microbe_gene.add_argument('--microbe_gene_file', help='Specify the file that contains microbe gene association.')
    parser_microbe_gene.add_argument('--gene_file', help='Specify the file that contains gene expression matrix.')
    parser_microbe_gene.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_microbe_gene.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_microbe_gene.add_argument('--output_prefix', help='Specify the output prefix of the saved model parameters and dataset.')
    parser_microbe_gene.add_argument('--microbe_corr_threshold', default=0.05, help='Specify the correlation threshold for microbe to be considered assocated to a specific gene. Default set to 0.05')
    parser_microbe_gene.add_argument('--batch_size', default=32, help='Specify the batch size during training. Default set to 32.')
    parser_microbe_gene.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during training. Default set to 64.')
    parser_microbe_gene.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during training. Default set to 0.005.')
    parser_microbe_gene.add_argument('--epochs', default=20, help='Specify the epochs during training. Default set to 20.')
    parser_microbe_gene.add_argument('--sample_number', default=0, help='Specify the number of samples you want to use during training. Useful for downsampling. Default set to 0 (no downsampling).')
    parser_microbe_gene.add_argument('--seed', default=42, help='Specify the seed you want to use during training. Default set to 42.')

    parser_microbe_SNP = subparsers.add_parser('microbe_SNP', help='Train the model with microbe and SNP data')
    parser_microbe_SNP.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_microbe_SNP.add_argument('--microbe_file', help='Specify the file that contains microbe matrix.')
    parser_microbe_SNP.add_argument('--SNP_file', help='Specify the file that contains SNP matrix.')
    parser_microbe_SNP.add_argument('--SNP_coding', help='Specify the file that contains associations between coding SNP and genes.')
    parser_microbe_SNP.add_argument('--SNP_noncoding', help='Specify the file that contains associations between noncoding SNP and genes.')
    parser_microbe_SNP.add_argument('--microbe_gene_file', help='Specify the file that contains microbe gene association.')
    parser_microbe_SNP.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_microbe_SNP.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_microbe_SNP.add_argument('--output_prefix', help='Specify the output prefix of the saved model parameters and dataset.')
    parser_microbe_SNP.add_argument('--microbe_corr_threshold', default=0.05, help='Specify the correlation threshold for microbe to be considered assocated to a specific gene. Default set to 0.05')
    parser_microbe_SNP.add_argument('--batch_size', default=32, help='Specify the batch size during training. Default set to 32.')
    parser_microbe_SNP.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during training. Default set to 64.')
    parser_microbe_SNP.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during training. Default set to 0.005.')
    parser_microbe_SNP.add_argument('--epochs', default=20, help='Specify the epochs during training. Default set to 20.')
    parser_microbe_SNP.add_argument('--sample_number', default=0, help='Specify the number of samples you want to use during training. Useful for downsampling. Default set to 0 (no downsampling).')
    parser_microbe_SNP.add_argument('--seed', default=42, help='Specify the seed you want to use during training. Default set to 42.')
    
    parser_gene_SNP = subparsers.add_parser('gene_SNP', help='Train the model with microbe and SNP data')
    parser_gene_SNP.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_gene_SNP.add_argument('--gene_file', help='Specify the file that contains gene expression matrix.')
    parser_gene_SNP.add_argument('--SNP_file', help='Specify the file that contains SNP matrix.')
    parser_gene_SNP.add_argument('--SNP_coding', help='Specify the file that contains associations between coding SNP and genes.')
    parser_gene_SNP.add_argument('--SNP_noncoding', help='Specify the file that contains associations between noncoding SNP and genes.')
    parser_gene_SNP.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_gene_SNP.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_gene_SNP.add_argument('--output_prefix', help='Specify the output prefix of the saved model parameters and dataset.')
    parser_gene_SNP.add_argument('--batch_size', default=32, help='Specify the batch size during training. Default set to 32.')
    parser_gene_SNP.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during training. Default set to 64.')
    parser_gene_SNP.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during training. Default set to 0.005.')
    parser_gene_SNP.add_argument('--epochs', default=20, help='Specify the epochs during training. Default set to 20.')
    parser_gene_SNP.add_argument('--sample_number', default=0, help='Specify the number of samples you want to use during training. Useful for downsampling. Default set to 0 (no downsampling).')
    parser_gene_SNP.add_argument('--seed', default=42, help='Specify the seed you want to use during training. Default set to 42.')

    parser_microbe_gene_SNP = subparsers.add_parser('microbe_gene_SNP', help='Train the model with microbe and SNP data')
    parser_microbe_gene_SNP.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_microbe_gene_SNP.add_argument('--microbe_file', help='Specify the file that contains microbe matrix.')
    parser_microbe_gene_SNP.add_argument('--gene_file', help='Specify the file that contains gene expression matrix.')
    parser_microbe_gene_SNP.add_argument('--SNP_file', help='Specify the file that contains SNP matrix.')
    parser_microbe_gene_SNP.add_argument('--SNP_coding', help='Specify the file that contains associations between coding SNP and genes.')
    parser_microbe_gene_SNP.add_argument('--SNP_noncoding', help='Specify the file that contains associations between noncoding SNP and genes.')
    parser_microbe_gene_SNP.add_argument('--microbe_gene_file', help='Specify the file that contains microbe gene association.')
    parser_microbe_gene_SNP.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_microbe_gene_SNP.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_microbe_gene_SNP.add_argument('--output_prefix', help='Specify the output prefix of the saved model parameters and dataset.')
    parser_microbe_gene_SNP.add_argument('--microbe_corr_threshold', default=0.05, help='Specify the correlation threshold for microbe to be considered assocated to a specific gene. Default set to 0.05')
    parser_microbe_gene_SNP.add_argument('--batch_size', default=32, help='Specify the batch size during training. Default set to 32.')
    parser_microbe_gene_SNP.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during training. Default set to 64.')
    parser_microbe_gene_SNP.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during training. Default set to 0.005.')
    parser_microbe_gene_SNP.add_argument('--epochs', default=20, help='Specify the epochs during training. Default set to 20.')
    parser_microbe_gene_SNP.add_argument('--sample_number', default=0, help='Specify the number of samples you want to use during training. Useful for downsampling. Default set to 0 (no downsampling).')
    parser_microbe_gene_SNP.add_argument('--seed', default=42, help='Specify the seed you want to use during training. Default set to 42.')

    parser_evaluation = subparsers.add_parser('evaluation', help='Evaluate the model with cross-validation, and plot AUC and AUPR curves')
    parser_evaluation.add_argument('--dataset_file', help='Specify the file that contains constructed torch_geometric dataset, which is saved as {output_prefix}_training.pt.')
    parser_evaluation.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_evaluation.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_evaluation.add_argument('--output_prefix', help='Specify the output prefix of the AUC and AUPR plot.')
    parser_evaluation.add_argument('--batch_size', default=32, help='Specify the batch size during cross-validation. Default set to 32.')
    parser_evaluation.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number during cross-validation. Default set to 64.')
    parser_evaluation.add_argument('--epochs', default=20, help='Specify the epochs during cross-validation. Default set to 20.')
    parser_evaluation.add_argument('--folds', default=5, help='Specify the folds for cross-validation. Default set to 5.')
    parser_evaluation.add_argument('--learning_rate', default=0.005, help='Specify the learning rate during cross-validation. Default set to 0.005.')
    parser_evaluation.add_argument('--sample_list', default=None, help='Specify the sample list pickle file if downsampling is used in training, which is saved as {output_prefix}_sample_index.pkl. Default set to None.')
    parser_evaluation.add_argument('--multi_class', action='store_true', help='Use this argument if the target class is multi-class.')
    parser_evaluation.add_argument('--seed', default=42, help='Specify the seed you want to use during cross-validation. Default set to 42.')

    parser_explanation = subparsers.add_parser('explanation', help='Explain the model with GraphSVX to generate importance for each feature and node')
    parser_explanation.add_argument('--node_file', help='Specify the file that contains constructed node information, which is saved as {output_prefix}_nodes.pt.')
    parser_explanation.add_argument('--meta_file', help='Specify the file that contains meta information.')
    parser_explanation.add_argument('--model_file', help='Specify the file that contains model information.')
    parser_explanation.add_argument('--sample_list', default=None, help='Specify the sample list pickle file if downsampling is used in training, which is saved as {output_prefix}_sample_index.pkl. Default set to None.')
    parser_explanation.add_argument('--pathway_file', help='Specify the file that contains pathway information.')
    parser_explanation.add_argument('--class_name', help='Specify the column name in the meta file that you would like the model to predict.')
    parser_explanation.add_argument('--output_prefix', help='Specify the output prefix of the explanation tensor.')
    parser_explanation.add_argument('--hidden_channels', default=64, help='Specify the hidden layer channel number of the input model. Default set to 64.')
    parser_explanation.add_argument('--multi_class', action='store_true', help='Use this argument if the target class is multi-class.')
    
    
    args = parser.parse_args()
    module_name = args.subparser_name
    
    if module_name == 'gene':
        pathway_file = args.pathway_file
        gene_file = args.gene_file
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        learning_rate = float(args.learning_rate)
        epochs = int(args.epochs)
        sample_num = int(args.sample_number)
        seed = int(args.seed)
        
        training_g(pathway_file, gene_file, meta_file, class_name, output_prefix, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed)
    
    elif module_name == 'microbe':
        pathway_file = args.pathway_file
        microbe_file = args.microbe_file
        microbe_gene_file = args.microbe_gene_file
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        microbe_corr_threshold = float(args.microbe_corr_threshold)
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        learning_rate = float(args.learning_rate)
        epochs = int(args.epochs)
        sample_num = int(args.sample_number)
        seed = int(args.seed)
        
        training_m(pathway_file, microbe_file, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed)
        
    elif module_name == 'microbe_gene':
        pathway_file = args.pathway_file
        microbe_file = args.microbe_file
        gene_file = args.gene_file
        microbe_gene_file = args.microbe_gene_file
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        microbe_corr_threshold = float(args.microbe_corr_threshold)
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        learning_rate = float(args.learning_rate)
        epochs = int(args.epochs)
        sample_num = int(args.sample_number)
        seed = int(args.seed)
        
        training_mg(pathway_file, microbe_file, gene_file, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed)
        
    elif module_name == 'microbe_SNP':
        pathway_file = args.pathway_file
        microbe_file = args.microbe_file
        SNP_file = args.SNP_file
        SNP_coding = args.SNP_coding
        SNP_noncoding = args.SNP_noncoding
        microbe_gene_file = args.microbe_gene_file
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        microbe_corr_threshold = float(args.microbe_corr_threshold)
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        learning_rate = float(args.learning_rate)
        epochs = int(args.epochs)
        sample_num = int(args.sample_number)
        seed = int(args.seed)
        
        training_msnp(pathway_file, microbe_file, SNP_file, SNP_coding, SNP_noncoding, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed)
        
    elif module_name == 'gene_SNP':
        pathway_file = args.pathway_file
        gene_file = args.gene_file
        SNP_file = args.SNP_file
        SNP_coding = args.SNP_coding
        SNP_noncoding = args.SNP_noncoding
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        learning_rate = float(args.learning_rate)
        epochs = int(args.epochs)
        sample_num = int(args.sample_number)
        seed = int(args.seed)
        
        training_gsnp(pathway_file, gene_file, SNP_file, SNP_coding, SNP_noncoding, meta_file, class_name, output_prefix, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed)
    
    elif module_name == 'microbe_gene_SNP':
        pathway_file = args.pathway_file
        microbe_file = args.microbe_file
        gene_file = args.gene_file
        SNP_file = args.SNP_file
        SNP_coding = args.SNP_coding
        SNP_noncoding = args.SNP_noncoding
        microbe_gene_file = args.microbe_gene_file
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        microbe_corr_threshold = float(args.microbe_corr_threshold)
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        learning_rate = float(args.learning_rate)
        epochs = int(args.epochs)
        sample_num = int(args.sample_number)
        seed = int(args.seed)
        
        training_mgsnp(pathway_file, microbe_file, gene_file, SNP_file, SNP_coding, SNP_noncoding, microbe_gene_file, meta_file, class_name, output_prefix, microbe_corr_threshold, batch_size, hidden_channels, learning_rate, epochs, sample_num, seed)
    
    elif module_name == 'evaluation':
        dataset_file = args.dataset_file
        meta_file = args.meta_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        batch_size = int(args.batch_size)
        hidden_channels = int(args.hidden_channels)
        epochs = int(args.epochs)
        folds = int(args.folds)
        learning_rate = float(args.learning_rate)
        multi_class = args.multi_class
        sample_list = args.sample_list
        seed = int(args.seed)
        
        evaluation(dataset_file, meta_file, class_name, output_prefix, batch_size, hidden_channels, epochs, folds, learning_rate, multi_class, sample_list, seed)

    elif module_name == 'explanation':
        node_file = args.node_file
        meta_file = args.meta_file
        model_file = args.model_file
        pathway_file = args.pathway_file
        class_name = args.class_name
        output_prefix = args.output_prefix
        hidden_channels = int(args.hidden_channels)
        multi_class = args.multi_class
        sample_list = args.sample_list
        
        explanation(node_file, meta_file, model_file, pathway_file, class_name, output_prefix, hidden_channels, multi_class, sample_list)

if __name__ == "__main__":
    main()
