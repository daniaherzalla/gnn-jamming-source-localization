import torch
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import MLP, GCN, GraphSAGE, GIN, GAT, PNA, AttentionalAggregation, global_mean_pool, global_max_pool, global_add_pool, ResGatedGraphConv
from torch_geometric.nn import GPSConv, GINEConv

from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv
from torch.nn import Linear, ModuleList, Dropout, Sequential
from torch_geometric.nn import GPSConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from utils import set_seeds_and_reproducibility
from config import params

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class GatedGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(GatedGCN, self).__init__()
        self.conv = ResGatedGraphConv(in_channels, out_channels, **kwargs)


class GraphGPS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads, dropout_rate, act='relu', norm='batch_norm', jk=None):
        super(GraphGPS, self).__init__()
        self.convs = ModuleList()
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.norms = ModuleList()
        self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()

        # Initial linear layer to transform input features to hidden_channels
        self.input_proj = Linear(in_channels, hidden_channels)

        # Define GPSConv layers
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            conv = GPSConv(hidden_channels, GINConv(nn), heads=heads, dropout=0.0)
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))  # Adjust for multi-head attention

        # Applying Jumping Knowledge
        if jk:
            self.jk = JumpingKnowledge(mode=jk, channels=hidden_channels, num_layers=num_layers)
            final_channels = hidden_channels * num_layers if jk == 'cat' else hidden_channels
        else:
            self.jk = None
            final_channels = hidden_channels

        self.dropout = Dropout(dropout_rate)
        self.lin = Linear(final_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        # Transform input features to hidden_channels
        x = self.input_proj(x)

        xs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)
            xs.append(x)

        if self.jk:
            x = self.jk(xs)
        else:
            x = xs[-1]

        x = global_max_pool(x, batch) if batch is not None else x
        x = self.lin(x)
        return x


class GNN(torch.nn.Module):
    """
    A GNN model for predicting jammer coordinates.

    Args:
        dropout_rate (float): The dropout rate for regularization.
        num_heads (int): The number of attention heads in the GAT layers.
        in_channels (int): Input features dimension: drone pos (x,y,z), RSSI, jamming status, distance to centroid.
    """
    def __init__(self, in_channels, dropout_rate=params['dropout_rate'], num_heads=params['num_heads'], model_type=params['model'], hidden_channels=params['hidden_channels'], out_channels=params['out_channels'], num_layers=params['num_layers'], out_features=params['out_features'], act='relu', norm=None, deg=None):
        super(GNN, self).__init__()

        # Model definitions
        print("model_type: ", model_type)
        if model_type == 'MLP':
            self.gnn = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type == 'GCN':
            self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type == 'Sage':
            self.gnn = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type == 'GIN':
            self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type in ['GAT', 'GATv2']:
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm, heads=num_heads, v2='v2' in model_type)
        elif model_type in 'GatedGCN':
            self.gnn = GatedGCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm) # check attn type from paper
        elif model_type == 'PNA':
            self.gnn = PNA(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers,aggregators=['mean', 'min', 'max', 'std'],scalers=['identity'], dropout=0.0, act=act, norm=None, deg=deg)
        elif model_type == 'GPS':
            self.gnn = GraphGPS(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, heads=num_heads, dropout_rate=0.0, act=act, norm='batch', jk=None) # 'cat'

        # Final layer
        self.attention_pool = AttentionalAggregation(gate_nn=Linear(out_channels, 1))
        self.regressor = Linear(out_channels, 3)
        self.classifier = Linear(out_channels, 2)  # Classifier layer, outputs two logits for binary classification
        self.dropout = torch.nn.Dropout(dropout_rate)
        # Initialize weights
        init_weights(self)

    def forward(self, data):
        """
        Forward pass for the GNN.

        Args:
            data (Data): The input data containing node features and edge indices.

        Returns:
            Tensor: The predicted coordinates of the jammer.
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # Handle based on the type of GNN
        if isinstance(self.gnn, GCN):
            # GCN supports edge weights
            x = self.gnn(x, edge_index, edge_weight=edge_weight)
        else:
            # Fallback for other types if no specific handling is needed
            x = self.gnn(x, edge_index)

        # Apply GNN layers
        if params['pooling'] == 'max':
            x = global_max_pool(x, data.batch)
        elif params['pooling'] == 'mean':
            x = global_mean_pool(x, data.batch)
        elif params['pooling'] == 'sum':
            x = global_add_pool(x, data.batch)
        elif params['pooling'] == 'att':
            x = self.attention_pool(x, data.batch)

        x = self.dropout(x)  # apply dropout last layer

        # Regression and classification outputs
        reg_output = self.regressor(x)
        class_output = self.classifier(x)

        print()

        return reg_output, class_output
