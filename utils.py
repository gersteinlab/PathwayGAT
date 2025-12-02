import pandas as pd
import numpy as np
from itertools import combinations
import torch
from torch_geometric.data import Data, Dataset
import random
import pickle
from types import SimpleNamespace

# Read pathway file and get genes related to pathway
def parse_pathway_file(filename):
    df = pd.read_csv(filename, sep='\t', header=None, names=['pathway', 'genes'])
    df['genes'] = df['genes'].str.split(',')
    
    unionGenes = set().union(*df['genes'])
    pathdict = df.set_index('pathway')['genes'].to_dict()
    
    return unionGenes, pathdict

# Calculate the adjacency between two pathways by the number of overlapping genes
def calculate_adjacency(pathdict):
    pathways = list(pathdict.keys())
    num_pathways = len(pathways)
    adjacency = np.zeros((num_pathways, num_pathways))
    
    for i, j in combinations(range(num_pathways), 2):
        overlap = len(set(pathdict[pathways[i]]) & set(pathdict[pathways[j]]))
        score = overlap / np.sqrt(len(pathdict[pathways[i]]) * len(pathdict[pathways[j]]))
        adjacency[i, j] = score
        adjacency[j, i] = score  # symmetric matrix
        
    return adjacency

# Read gene expression from files
def parse_gene_expression(filename, unionGenes, sample_list):
    tcgaExpr = pd.read_csv(filename, sep='\t', index_col=0)
    overlapping_genes = list(unionGenes.intersection(tcgaExpr.columns))
    subset_expr = tcgaExpr[overlapping_genes]
    if sample_list is not None:
        subset_expr = subset_expr.iloc[sample_list, ]
    
    return subset_expr, overlapping_genes

# Construct torch nodes for each pathway
def construct_pathnodes(pathdict, subset_expr):
    pathways = list(pathdict.keys())
    genes = subset_expr.columns
    pathnodes_list = []

    for pathway in pathways:
        mask = genes.isin(pathdict[pathway]).astype(int) #.values  # Create a binary mask for genes in the pathway
        pathway_data = subset_expr * mask  # Multiply each row (sample) in subset_expr by the mask
        pathnodes_list.append(pathway_data.values)

    return torch.tensor(pathnodes_list)

# Read the file containing microbe-gene correlation
def read_microbe_gene_corr(file_path, column_selector=2):
    # Read the file into a DataFrame with tab as the delimiter
    df = pd.read_csv(file_path, header=0, delimiter='\t')
    
    # Check if the DataFrame has the expected number of columns
    if df.shape[1] <= column_selector:
        raise ValueError(f"Expected at least {column_selector + 1} columns in the file, but found only {df.shape[1]} columns.")
    
    # Pivot the DataFrame to reshape it into the desired matrix format
    microbe_gene_corr = df.pivot(index='microbe', columns='gene', values=df.columns[column_selector])
    
    return microbe_gene_corr

# Generate the microbe features that can be added to the node
def generate_microbe_features(microbe_abundance, microbe_gene_corr, cutoff, pathdict):
    # Convert the columns of the microbe_gene_corr matrix to a set for faster intersection operations
    microbe_gene_corr_columns_set = set(microbe_gene_corr.columns)
    
    microbe_features = {}
    for pathway, genes in pathdict.items():
        # Use set intersection to find the overlapping genes
        overlapping_genes = list(set(genes) & microbe_gene_corr_columns_set)
        
        # If no overlapping genes, continue to the next pathway
        if not overlapping_genes:
            continue
        
        # Filter the correlation matrix for overlapping genes
        filtered_corr = microbe_gene_corr[overlapping_genes].dropna(how='all')
        
        # Create a boolean mask for microbes correlated with genes in the pathway
        mask = (filtered_corr.abs() > cutoff).any(axis=1)
        
        # Set the abundance of microbes which are 'False' in the mask to 0
        pathway_abundance = microbe_abundance.copy()
        pathway_abundance[mask[~mask].index] = 0 # Modified by Weihao to avoid bug
                
        # Store the pathway abundance matrix in the dictionary
        microbe_features[pathway] = pathway_abundance

    return microbe_features

# Read the file containing SNP-gene linkage
def parse_snp_gene_link(file):
    snp_to_genes = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            snp = parts[0]
            genes = parts[1:]
            if snp in snp_to_genes:
                snp_to_genes[snp].extend(genes)
            else:
                snp_to_genes[snp] = genes
    return snp_to_genes

# Construct input SNP dataset
def snp_info(cdlinkfin, nclinkfin, snp_sample_fin, wpdict, sample_list):
    # Parse SNP to gene mappings for both coding and noncoding files
    cd_snp_to_genes = parse_snp_gene_link(cdlinkfin)
    nc_snp_to_genes = parse_snp_gene_link(nclinkfin)
    snp_to_genes = {**cd_snp_to_genes, **nc_snp_to_genes}
    
    # Read SNP-sample matrix using pandas
    snp_sample_df = pd.read_csv(snp_sample_fin, sep='\t', index_col=0)
    # Added by Weihao to downsample
    snp_sample_df = snp_sample_df.iloc[sample_list, ]
    snp_sample_df = snp_sample_df.T
    
    # Create an output dictionary
    snpdict = {}
    
    for pathway, genes in wpdict.items():
        # Create a boolean mask for SNPs based on their linked genes being in the current pathway
        mask = snp_sample_df.index.to_series().apply(lambda snp: any(gene in genes for gene in snp_to_genes.get(snp, [])))
        # Apply the mask to the dataframe to update values
        filtered_df = snp_sample_df.copy()
        filtered_df[~mask] = 0
        
        # Transpose the dataframe for sample by SNP and store it in the dictionary
        snpdict[pathway] = filtered_df.T

    return snpdict

# Construct pathway node with only gene expression data
def construct_pathnodes1g(pathdict, subset_expr):
    pathways = list(pathdict.keys())
    genes = subset_expr.columns
    pathnodes_list = []

    for pathway in pathways:
        mask = genes.isin(pathdict[pathway]).astype(int)
        pathway_data_expr = subset_expr * mask
        pathnodes_list.append(pathway_data_expr.values)

    return torch.tensor(pathnodes_list)


# Construct pathway node with only microbe data
def construct_pathnodes1m(pathdict, microbe_features):
    pathways = list(pathdict.keys())
    
    pathnodes_list = []

    for pathway in pathways:
        pathway_data_microbe = microbe_features[pathway].values
        pathnodes_list.append(pathway_data_microbe)

    return torch.tensor(pathnodes_list)

# Construct pathway node with microbe and gene expression data
def construct_pathnodes2(pathdict, subset_expr, microbe_features):
    pathways = list(pathdict.keys())
    genes = subset_expr.columns
    pathnodes_list = []

    for pathway in pathways:
        mask = genes.isin(pathdict[pathway]).astype(int)
        pathway_data_expr = subset_expr * mask
        pathway_data_microbe = microbe_features[pathway].values
        pathway_data = np.concatenate([pathway_data_expr.values, pathway_data_microbe], axis=1)
        pathnodes_list.append(pathway_data)

    return torch.tensor(pathnodes_list)

# Construct pathway node with microbe and SNP data
def construct_pathnodes3msnp(pathdict, microbe_features, snp_dict):
    pathways = list(pathdict.keys())
   
    pathnodes_list = []

    for pathway in pathways:

        pathway_data_microbe = microbe_features[pathway]
        pathway_data_snp = snp_dict[pathway]
        common_idx = pathway_data_microbe.index.intersection(pathway_data_snp.index)

        pathway_data = np.concatenate([pathway_data_microbe.loc[common_idx].values, pathway_data_snp.loc[common_idx].values], axis=1)
        pathnodes_list.append(pathway_data)

    return torch.tensor(pathnodes_list)

# Construct pathway node with gene and SNP data
def construct_pathnodes3gsnp(pathdict, subset_expr, snp_dict):
    pathways = list(pathdict.keys())
    genes = subset_expr.columns
    pathnodes_list = []

    for pathway in pathways:
        mask = genes.isin(pathdict[pathway]).astype(int)
        pathway_data_expr = subset_expr * mask
        pathway_data_snp = snp_dict[pathway]
        
        common_idx = pathway_data_expr.index.intersection(pathway_data_snp.index)
        pathway_data = np.concatenate([pathway_data_expr.loc[common_idx].values, pathway_data_snp.loc[common_idx].values], axis=1)
        pathnodes_list.append(pathway_data)

    return torch.tensor(pathnodes_list)

# Construct pathway node with microbe, gene expression, and SNP data
def construct_pathnodes4(pathdict, subset_expr, microbe_features, snp_dict):
    pathways = list(pathdict.keys())
    genes = subset_expr.columns
    pathnodes_list = []

    for pathway in pathways:
        mask = genes.isin(pathdict[pathway]).astype(int)
        pathway_data_expr = subset_expr * mask
        pathway_data_microbe = microbe_features[pathway] #.values
        pathway_data_snp = snp_dict[pathway] #.values
        
        common_idx = pathway_data_microbe.index.intersection(pathway_data_snp.index).intersection(subset_expr.index)
        pathway_data = np.concatenate([pathway_data_expr.loc[common_idx].values, pathway_data_microbe.loc[common_idx].values, pathway_data_snp.loc[common_idx].values], axis=1)
        pathnodes_list.append(pathway_data)

    return torch.tensor(pathnodes_list)

# Convert adjacency to edge index for torch_geometric
def adjacency_to_edge_index(adjacency):
    source, target = np.where(adjacency > 0)
    return torch.tensor([source, target], dtype=torch.long)

# Generate dataset that can be accepted by torch_geometric
def create_geometric_dataset(pathnodes, edge_index, labels):
    dataset = []
    n_pathways, n_samples, m_genes = pathnodes.shape
            
    for i in range(n_samples):
        # Extract the node information for the current sample across all pathways
        x = pathnodes[:, i, :]
        
        # Create a PyTorch Geometric data object for the current sample
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(labels[i], dtype=torch.long))
        dataset.append(data)
    
    return dataset

# Split the dataset into given length (used in training dataset split)
def random_split(dataset, lengths):
    # Determine the indices for the split
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    current = 0
    splits = []
    for length in lengths:
        split_indices = indices[current:current+length]
        splits.append([dataset[i] for i in split_indices])
        current += length
    return splits
