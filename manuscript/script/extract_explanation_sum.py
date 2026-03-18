import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model import PathwayGAT3
from src.utils import *
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

output_prefix = 'JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35'

explanation_list = torch.load(f'{output_prefix}_explanation_GNNExplainer.pt', map_location=torch.device('cpu'))
for idx, explanation in enumerate(explanation_list):
    torch.save(explanation, f'{output_prefix}_explanation_GNNExplainer_split_{idx}.pt')

total_gene_list = np.zeros((248, 6968))
total_microbe_list = np.zeros((248, 715))
total_node_list = np.zeros((248, 622))
total_node_gene_list = np.zeros((248, 622))
total_node_microbe_list = np.zeros((248, 622))
total_node_feature_list = np.zeros((622, 7683))

for i in range(248):
    explanation_GNNE = torch.load(f'{output_prefix}_explanation_GNNExplainer_split_{i}.pt', map_location=torch.device('cpu'))
    total_gene_list[i, :] = np.array(torch.sum(explanation_GNNE['node_mask'][:, :6968], axis=0))
    total_microbe_list[i, :] = np.array(torch.sum(explanation_GNNE['node_mask'][:, 6968:], axis=0))
    total_node_list[i, :] = np.array(torch.sum(explanation_GNNE['node_mask'], axis=1))
    total_node_gene_list[i, :] = np.array(torch.sum(explanation_GNNE['node_mask'][:, :6968], axis=1))
    total_node_microbe_list[i, :] = np.array(torch.sum(explanation_GNNE['node_mask'][:, 6968:], axis=1))
    total_node_feature_list = total_node_feature_list + explanation_GNNE['node_mask'].numpy()

microbe_table = pd.read_csv('../data/TCGA_COAD.microbe.txt', sep='\t', header=0, index_col=0)
meta_table = pd.read_csv('../data/COAD_sample_metadata.processed.txt', sep='\t', header=0, index_col=0)
sample_microbe_table = microbe_table.reindex(meta_table.index)
gene_table = pd.read_csv('../data/TCGA_COAD.gene.txt', sep='\t', header=0, index_col=0)
sample_gene_table = gene_table.reindex(meta_table.index)
node_table = pd.read_csv('../data/wikipath2021.tsv', sep='\t', header=None, names=['pathway', 'gene'])
total_gene_data = pd.DataFrame(total_gene_list, index=sample_gene_table.index, columns=sample_gene_table.columns)
total_gene_data.to_csv(f'{output_prefix}_explanation_gene.txt', sep='\t', header=True, index=True, index_label='sample')
total_microbe_data = pd.DataFrame(total_microbe_list, index=sample_microbe_table.index, columns=sample_microbe_table.columns)
total_microbe_data.to_csv(f'{output_prefix}_explanation_microbe.txt', sep='\t', header=True, index=True, index_label='sample')
total_node_data = pd.DataFrame(total_node_list, index=sample_microbe_table.index, columns=node_table['pathway'])
total_node_data.to_csv(f'{output_prefix}_explanation_pathway.txt', sep='\t', header=True, index=True, index_label='sample')
total_node_gene_data = pd.DataFrame(total_node_gene_list, index=sample_microbe_table.index, columns=node_table['pathway'])
total_node_gene_data.to_csv(f'{output_prefix}_explanation_gene_pathway.txt', sep='\t', header=True, index=True, index_label='sample')
total_node_microbe_data = pd.DataFrame(total_node_microbe_list, index=sample_microbe_table.index, columns=node_table['pathway'])
total_node_microbe_data.to_csv(f'{output_prefix}_explanation_microbe_pathway.txt', sep='\t', header=True, index=True, index_label='sample')
total_node_feature_data = pd.DataFrame(total_node_feature_list, index=node_table['pathway'], columns=list(gene_table.columns) + list(microbe_table.columns))
total_node_feature_data.to_csv(f'{output_prefix}_explanation_pathway_feature.txt', sep='\t', header=True, index=True, index_label='pathway')
