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
            self.output_act_radius = torch.nn.Sigmoid()  # For radius, ensure outputs are [0, 1]
            self.output_act_angles = torch.nn.Tanh()  # For angles, handle outputs in [-1, 1]
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
        x = self.attention_pool(x, data.batch)  # Apply attention pooling to get a single vector for the graph
        # x = global_mean_pool(x, data.batch)  # Pooling to predict a single output per graph
        x = self.dropout(x)  # apply dropout last layer
        # print("x: ", x)
        if params['feats'] == 'cartesian':
            x = self.output_act(self.regressor(x))  # Predict the jammer's coordinates
        elif params['feats'] == 'polar':
            # Apply sigmoid activation to r
            x_all = self.regressor(x)
            x_radius = self.output_act_radius(x_all[:, 0].unsqueeze(1))  # Radius: first feature, reshaped to keep two dimensions
            x_angles = self.output_act_angles(x_all[:, 1:])  # Angles: remaining features

            # Do Not apply activation to r
            # x_all = self.regressor(x)
            # x_radius = x_all[:, 0].unsqueeze(1)  # Directly use the output without activation for radius
            # x_angles = torch.tanh(x_all[:, 1:])  # Apply tanh for angle components

            # print(f"x_radius shape: {x_radius.shape}")
            # print(f"x_angles shape: {x_angles.shape}")

            x = torch.cat((x_radius, x_angles), dim=1)  # Combine back into one tensor
        return x

    def print_weights(self):
        """
        Print the weights of the model.
        """
        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



# import torch
# from torch_geometric.graphgym import init_weights
# from torch_geometric.nn import global_mean_pool, GATConv, AttentionalAggregation
# import torch.nn.functional as F
# from torch.nn import init
# from torch.nn import Linear
# from utils import set_seeds_and_reproducibility
# from config import params
#
# set_seeds_and_reproducibility()
#
#
# class GNN(torch.nn.Module):
#     """
#     A GNN model for predicting jammer coordinates.
#
#     Args:
#         num_heads_first_two (int): Number of attention heads in the first two layers.
#         num_heads_final (int): Number of attention heads in the final layer.
#         features_per_head_first_two (int): Number of features per head in the first two layers.
#         features_per_head_final (int): Number of features per head in the final layer.
#         in_channels (int): Input features dimension.
#     """
#     def __init__(self, num_heads_first_two=4, num_heads_final=6, features_per_head_first_two=256, features_per_head_final=121, in_channels=params['in_channels'], out_features=3):
#         super(GNN, self).__init__()
#
#         # Define the output dimension of the first GAT layer
#         # First two layers
#         self.gnn1 = GATConv(in_channels=in_channels, out_channels=features_per_head_first_two, heads=num_heads_first_two, dropout=0.0)
#         self.transform1 = Linear(in_channels, num_heads_first_two * features_per_head_first_two)  # Updated dimension
#         self.gnn2 = GATConv(in_channels=num_heads_first_two * features_per_head_first_two, out_channels=features_per_head_first_two, heads=num_heads_first_two, dropout=0.0)
#
#         # Final layer
#         self.gnn3 = GATConv(in_channels=num_heads_first_two * features_per_head_first_two, out_channels=features_per_head_final, heads=num_heads_final, dropout=0.0)
#
#         # Attention pooling
#         self.attention_pool = AttentionalAggregation(gate_nn=Linear(num_heads_final * features_per_head_final, 1))
#
#         # Final layer to predict output
#         self.regressor = Linear(num_heads_final * features_per_head_final, out_features)
#
#         self.dropout = torch.nn.Dropout(0.2)
#
#         # Initialize weights
#         init_weights(self)
#
#     def forward(self, data):
#         """
#         Forward pass for the GNN.
#
#         Args:
#             data (Data): The input data containing node features and edge indices.
#
#         Returns:
#             Tensor: The predicted coordinates of the jammer.
#         """
#         x, edge_index = data.x, data.edge_index
#
#         # Apply GNN layers with skip connections
#         # this one
#         x1 = self.gnn1(x, edge_index)
#         x1 = F.elu(x1)  # Skip connection with transformation
#         x2 = self.gnn2(x1, edge_index)
#         x2 = F.elu(x2 + self.transform1(x))  # Skip connection and ELU nonlinearity
#         x3 = self.gnn3(x2, edge_index)  # Outputs are automatically averaged because concat=False
#         x3 = self.attention_pool(x3, data.batch)  # Apply attention pooling to get a single vector for the graph
#         x3 = self.dropout(x3)  # apply dropout last layer
#         x3 = self.regressor(x3)  # Predict the jammer's coordinates
#         return x3
#
#         # # Apply GNN layers with skip connections
#         # x1 = self.gnn1(x, edge_index)
#         # x1 = F.elu(x1 + self.transform1(x))  # Skip connection with transformation
#         # x2 = self.gnn2(x1, edge_index)
#         # x2 = F.elu(x2 + x1)  # Skip connection and ELU nonlinearity
#         # x3 = self.gnn3(x2, edge_index)  # Outputs are automatically averaged because concat=False
#         # x3 = self.attention_pool(x3, data.batch)  # Apply attention pooling to get a single vector for the graph
#         # x3 = self.regressor(x3)  # Predict the jammer's coordinates
#         # return x3
#
#         # # Apply GNN layers
#         # x = self.gnn1(x, edge_index)
#         # x = F.elu(x)  # Apply ELU nonlinearity after the first GATConv layer
#         # x = self.gnn2(x, edge_index)
#         # x = F.elu(x)  # Apply ELU nonlinearity after the second GATConv layer
#         # x = self.gnn3(x, edge_index)
#         # x = self.attention_pool(x, data.batch)  # Apply attention pooling to get a single vector for the graph
#         # # x = self.dropout(x)  # apply dropout last layer
#         # x = self.regressor(x)  # Predict the jammer's coordinates
#         # return x
