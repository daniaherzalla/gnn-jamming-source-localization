import torch
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import MLP, GCN, GraphSAGE, GIN, GAT, AttentionalAggregation, global_mean_pool

from torch.nn import Linear
from utils import set_seeds_and_reproducibility
from config import params

class GNN(torch.nn.Module):
    """
    A GNN model for predicting jammer coordinates.

    Args:
        dropout_rate (float): The dropout rate for regularization.
        num_heads (int): The number of attention heads in the GAT layers.
        in_channels (int): Input features dimension: drone pos (x,y,z), RSSI, jamming status, distance to centroid.
    """
    def __init__(self, in_channels, dropout_rate=params['dropout_rate'], num_heads=params['num_heads'], model_type=params['model'], hidden_channels=params['hidden_channels'], out_channels=params['out_channels'], num_layers=params['num_layers'], out_features=params['out_features'], act='relu', norm=None):
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
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act='relu', norm=norm, heads=num_heads, v2='v2' in model_type)

        # Final layer
        self.attention_pool = AttentionalAggregation(gate_nn=Linear(out_channels, 1))
        self.regressor = Linear(out_channels, out_features)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output_act_tanh = torch.nn.Tanh()
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
        x, edge_index = data.x, data.edge_index

        # Apply GNN layers
        x = self.gnn(x, edge_index)
        # x = global_mean_pool(x, data.batch)
        x = self.attention_pool(x, data.batch)  # Apply attention pooling to get a single vector for the graph
        x = self.dropout(x)  # apply dropout last layer
        x = self.regressor(x)
        if params['activation']:
            x = self.output_act_tanh(x)
        return x
