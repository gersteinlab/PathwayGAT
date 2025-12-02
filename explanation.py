import torch
import numpy as np
import pandas as pd
from model import PathwayGAT2
from utils import *
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def explanation(node_file, meta_file, model_file, pathway_file, class_name, output_prefix, hidden_channels, multi_class, sample_list_dir):
    # Create dataset and target
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

    # Load model from training steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_file, map_location=device)
    model = PathwayGAT2(num_features=nodes.shape[2], hidden_channels=hidden_channels, num_classes=len(set(label_df['label'])), num_nodes=nodes.shape[0]).to(device)
    model.load_state_dict(state_dict)

    # Explain the model with GNNExplainer
    if multi_class:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
    else:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
    explanation_list = []
    for data in data_loader:
        data = data.to(device)
        data.x = data.x.to(torch.float32)
        explanation_list.append(explainer(data.x, data.edge_index, batch=data.batch))

    torch.save(explanation_list, f'{output_prefix}_explanation_GNNExplainer_split.pt')
