import torch
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import MLP, GCN, GraphSAGE, GIN, GAT, PNA, AttentionalAggregation, global_mean_pool, global_max_pool, global_add_pool, ResGatedGraphConv
from torch_geometric.nn import GPSConv, GINEConv


from torch.nn import Linear
from utils import set_seeds_and_reproducibility
from config import params

# from sem_gcn import SemGCN

# from sem_gcn_mdn import SemGCN_MDN_Graph


import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class GatedGCN(GCN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs):
        return ResGatedGraphConv(in_channels, out_channels, **kwargs)


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
            self.gnn = GatedGCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm, attn_type="multihead") # check attn type from paper
        elif model_type == 'PNA':
            self.gnn = PNA(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers,aggregators=['mean', 'min', 'max', 'std'],scalers=['identity'], dropout=0.0, act=act, norm=None, jk="lstm", deg=deg)
        elif model_type == 'GPS':
            # Initialize GPSConv specific layers
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                nn = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                )
                conv = GPSConv(hidden_channels, GINEConv(nn), heads=num_heads, dropout=dropout_rate)
                self.convs.append(conv)

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

        if hasattr(self, 'convs'):  # Check if using GPSConv
            for conv in self.convs:
                x = conv(x, edge_index)
        else:
            x = self.gnn(x, edge_index)  # Fallback to previous GNN model logic

        # Apply GNN layers
        if params['pooling'] == 'max':
            x = global_max_pool(x, data.batch)
        elif params['pooling'] == 'mean':
            x = global_mean_pool(x, data.batch)
        elif params['pooling'] == 'sum':
            x = global_add_pool(x, data.batch)
        elif params['pooling'] == 'att':
            x = self.attention_pool(x, data.batch)
        elif params['pooling'] == 'concat_mean_max':
            x_max = global_max_pool(x, data.batch)
            x_mean = global_mean_pool(x, data.batch)
            x = torch.cat((x_max, x_mean), dim=1)  # Concatenate along the feature dimension

        x = self.dropout(x)  # apply dropout last layer
        x = self.regressor(x)
        if params['activation']:
            x = self.output_act_tanh(x)
        return x
