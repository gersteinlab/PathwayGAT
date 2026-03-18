import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model import PathwayGAT3
from src.utils import *
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

node_file = 'JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_nodes.pt' 
meta_file = '../data/COAD_sample_metadata.processed.txt'
model_file = 'JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35.final_model.pth'
pathway_file = '../data/wikipath2021.tsv'
class_name = 'sample_type2'
output_prefix = 'JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35'
hidden_channels = 128
multi_class = False
sample_list_dir = 'JHU_COAD_microbe_q75m2_gene_q75m2_sampleType_0.15_32_128_0.001_35_sample_index.pkl'

nodes = torch.load(node_file)
nodes = nodes.to(torch.float32)
label_df = pd.read_csv(meta_file, header=0, delimiter='\t', index_col=0)
label_df['label'], _ = pd.factorize(label_df[class_name])
if sample_list_dir is not None:
    with open(sample_list_dir, 'rb') as f:
        sample_list = pickle.load(f)
    label_df = label_df.iloc[sample_list, ]

wpgene, wpdict = parse_pathway_file(pathway_file)
wpadj = calculate_adjacency(wpdict)
wp_edge = adjacency_to_edge_index(wpadj)

dataset = create_geometric_dataset(nodes, wp_edge, label_df['label'])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(model_file, map_location=device)
model = PathwayGAT3(num_features=nodes.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label'])), num_nodes=nodes.shape[0]).to(device)
model.load_state_dict(state_dict)

model.eval()
with torch.no_grad():
    for data in data_loader:    
        data = data.to(device)
        data.x = data.x.to(torch.float32)
        out1, (edge_idx1, attn1) = model.gat1(data.x, data.edge_index, return_attention_weights=True)
        out2, (edge_idx2, attn2) = model.gat2(out1, data.edge_index, return_attention_weights=True)

d_attention1 = {'source': [], 'destination': [], 'attention': []}
for i, (src, dst) in enumerate(edge_idx1.t().tolist()):
    d_attention1['source'].append(src)
    d_attention1['destination'].append(dst)
    d_attention1['attention'].append(attn1[i].tolist()[0])
d_attention2 = {'source': [], 'destination': [], 'attention': []}
for i, (src, dst) in enumerate(edge_idx2.t().tolist()):
    d_attention2['source'].append(src)
    d_attention2['destination'].append(dst)
    d_attention2['attention'].append(attn2[i].tolist()[0])

df_attention1 = pd.DataFrame(d_attention1)
df_attention2 = pd.DataFrame(d_attention2)
df_attention1.to_csv('attention/JHU_COAD_final_35_attention1.txt', sep='\t', header=True, index=False)
df_attention2.to_csv('attention/JHU_COAD_final_35_attention2.txt', sep='\t', header=True, index=False)
