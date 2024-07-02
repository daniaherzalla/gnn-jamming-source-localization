import torch
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import global_mean_pool, MLP, GCN, GraphSAGE, GIN, GAT, AttentionalAggregation

from torch.nn import Linear
from utils import set_seeds_and_reproducibility
from config import params

set_seeds_and_reproducibility()


class GNN(torch.nn.Module):
    """
    A GNN model for predicting jammer coordinates.

    Args:
        dropout_rate (float): The dropout rate for regularization.
        num_heads (int): The number of attention heads in the GAT layers.
        in_channels (int): Input features dimension: drone pos (x,y,z), RSSI, jamming status, distance to centroid.
    """
    def __init__(self, dropout_rate=params['dropout_rate'], num_heads=params['num_heads'], model_type=params['model'], in_channels=params['in_channels'], hidden_channels=params['hidden_channels'], out_channels=params['out_channels'], num_layers=params['num_layers'], out_features=params['out_features'], act='relu', norm=None):
        super(GNN, self).__init__()

        # Model definitions
        if model_type == 'MLP':
            self.gnn = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
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
        if params['feats'] == 'cartesian':
            self.output_act = torch.nn.Tanh()
        elif params['feats'] == 'polar':
            self.output_act = torch.nn.Sigmoid()  # ensure outputs are [0, 1]
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
        # print('data.x: ', data.x)

        # Apply GNN layers
        x = self.gnn(x, edge_index)
        x = self.attention_pool(x, data.batch)  # Apply attention pooling to get a single vector for the graph
        # x = global_mean_pool(x, data.batch)  # Pooling to predict a single output per graph
        x = self.dropout(x)  # apply dropout last layer
        # print("x: ", x)
        if params['feats'] == 'cartesian':
            x = self.output_act(self.regressor(x))  # Predict the jammer's coordinates
        elif params['feats'] == 'polar':
            # Apply sigmoid activation to r
            x_all = self.regressor(x)

            # Apply different activations to radius and angles
            x_radius = self.output_act(x_all[:, 0].unsqueeze(1))  # Radius
            x_theta = 2 * torch.pi * self.output_act(x_all[:, 1].unsqueeze(1))  # Theta, Azimuthal angle [0, 2π]
            if params['3d']:
                x_phi = torch.pi * self.output_act(x_all[:, 2].unsqueeze(1))  # Phi, Polar angle [0, π]
                x = torch.cat((x_radius, x_theta, x_phi), dim=1)
            else:
                x = torch.cat((x_radius, x_theta), dim=1)
        return x

    def print_weights(self):
        """
        Print the weights of the model.
        """
        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
