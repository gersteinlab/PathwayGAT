import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from utils import *

# Added from Shaoke

# Model used in the training process
class PathwayGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, pooling=None, k_ratio=1.0):
        super(PathwayGAT, self).__init__()
        self.pooling = pooling
        self.k_ratio = k_ratio
        self.gat1 = GATConv(num_features, hidden_channels, heads=1)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.reduce_dim = torch.nn.Linear(hidden_channels, 1)  # Reduce feature dimension from 64 to 1
        self.num_classes = num_classes
        outprefix="nopooling" if pooling is None else pooling
        if not (pooling is None or pooling == 'topk'):
            print(f'Error, the pooling should be topk or None')

        # The exact input dimension for the classifier will be determined in the forward pass
        self.classifier = None

    def forward(self, x, edge_index, batch, device):
        num_nodes = x.size(1)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        x = self.reduce_dim(x)
        
        # Flatten the tensor for no pooling or topK pooling
        if self.pooling == "topk":
            k = int(self.k_ratio * num_nodes)
            x, _ = topk_pool(x, batch, ratio=k)
            x = x.view(x.size(0), -1)
        elif self.pooling is None:
            x = x.view(batch.max().item() + 1, -1) # nodes 622 x batch 32 x hidden 
        
        # Dynamically define the classifier if it's None
        if self.classifier is None:
            classifier_input_dim = x.size(1)
            self.classifier = torch.nn.Linear(classifier_input_dim, self.num_classes).to(device) # Added device by Weihao to keep classifier on the same device as the rest of model and data
        
        return self.classifier(x)

# Models used in GraphSVX explanation
class LinearRegressionModel(torch.nn.Module):
    """Construct a simple linear regression

    """    
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred =self.linear1(x)
        return y_pred


class PathwayGAT2(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes, num_classes, pooling=None, k_ratio=1.0):
        super(PathwayGAT2, self).__init__()
        self.pooling = pooling
        self.k_ratio = k_ratio
        self.gat1 = GATConv(num_features, hidden_channels, heads=1)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.reduce_dim = torch.nn.Linear(hidden_channels, 1)  # Reduce feature dimension from 64 to 1
        self.num_classes = num_classes
        outprefix="nopooling" if pooling is None else pooling
        if not (pooling is None or pooling == 'topk'):
            print(f'Error, the pooling should be topk or None')

        self.classifier = torch.nn.Linear(num_nodes, self.num_classes)

    def forward(self, x, edge_index):
        edge_index = adjacency_to_edge_index(edge_index)
        
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        x = self.reduce_dim(x)   
        x = x.view(1, -1)
        
        return self.classifier(x)

# Model used in cross-validation
class PathwayGAT3(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes, num_classes, pooling=None, k_ratio=1.0):
        super(PathwayGAT3, self).__init__()
        self.pooling = pooling
        self.k_ratio = k_ratio
        self.gat1 = GATConv(num_features, hidden_channels, heads=1)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.reduce_dim = torch.nn.Linear(hidden_channels, 1)  # Reduce feature dimension from 64 to 1
        self.num_classes = num_classes
        outprefix="nopooling" if pooling is None else pooling
        if not (pooling is None or pooling == 'topk'):
            print(f'Error, the pooling should be topk or None')

        self.classifier = torch.nn.Linear(num_nodes, self.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        x = self.reduce_dim(x)   
        x = x.view(batch.max().item() + 1, -1)
        
        return self.classifier(x)