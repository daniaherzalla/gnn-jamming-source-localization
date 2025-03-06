import torch
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import MLP, GCN, GraphSAGE, GIN, GAT, PNA, AttentionalAggregation, global_mean_pool, global_max_pool, global_add_pool, ResGatedGraphConv
from torch_geometric.nn import GPSConv, GINEConv

from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GCNConv, GatedGraphConv, ResGatedGraphConv, PNAConv
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


# class GraphGPS(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads, dropout_rate, act='relu', jk='max', deg=None):
#         super(GraphGPS, self).__init__()
#         self.convs = ModuleList()
#         self.heads = heads
#         self.dropout_rate = dropout_rate
#         self.norms = ModuleList()
#         self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()
#
#         # Initial linear layer to transform input features to hidden_channels
#         self.input_proj = Linear(in_channels, hidden_channels)
#
#         # Define GPSConv layers
#         for _ in range(num_layers):
#             nn = Sequential(
#                 Linear(hidden_channels, hidden_channels),
#                 torch.nn.ReLU(),
#                 Linear(hidden_channels, hidden_channels),
#             )
#             conv = GPSConv(hidden_channels, ResGatedGraphConv(hidden_channels, hidden_channels, deg=deg), heads=heads, dropout=0.0, attn_type='performer')
#             self.convs.append(conv)
#             self.norms.append(BatchNorm(hidden_channels))  # Adjust for multi-head attention
#
#         # Applying Jumping Knowledge
#         if jk:
#             self.jk = JumpingKnowledge(mode=jk, channels=hidden_channels, num_layers=num_layers)
#             final_channels = hidden_channels * num_layers if jk == 'cat' else hidden_channels
#         else:
#             self.jk = None
#             final_channels = hidden_channels
#
#         self.dropout = Dropout(0.0)
#         self.lin = Linear(final_channels, out_channels)
#
#     def forward(self, x, edge_index, batch=None):
#         # Transform input features to hidden_channels
#         x = self.input_proj(x)
#
#         xs = []
#         for conv, norm in zip(self.convs, self.norms):
#             x = conv(x, edge_index)
#             x = norm(x)
#             x = self.act(x)
#             x = self.dropout(x)
#             xs.append(x)
#
#         if self.jk:
#             x = self.jk(xs)
#         else:
#             x = xs[-1]
#
#         x = global_max_pool(x, batch) if batch is not None else x
#         x = self.dropout(x)  # apply dropout last layer
#         x = self.lin(x)
#         return x


# class GraphGPS(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers, out_channels, heads, dropout_rate, act='relu', jk=None, deg=None):
#         super(GraphGPS, self).__init__()
#         self.convs = ModuleList()
#         self.heads = heads
#         self.dropout_rate = dropout_rate
#         self.norms = ModuleList()
#         self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()
#
#         # Initial linear layer to transform input features to hidden_channels
#         self.input_proj = Linear(in_channels, hidden_channels)
#
#         # Define GPSConv layers
#         for _ in range(num_layers):
#             nn = Sequential(
#                 Linear(hidden_channels, hidden_channels),
#                 torch.nn.ReLU(),
#                 Linear(hidden_channels, hidden_channels),
#             )
#             conv = GPSConv(hidden_channels, PNAConv(hidden_channels, hidden_channels, deg=deg, aggregators=['mean', 'max', 'std'], scalers=['identity']), heads=heads, attn_type='multihead', dropout=0.0)
#             self.convs.append(conv)
#             self.norms.append(BatchNorm(hidden_channels))  # Adjust for multi-head attention
#
#         # Applying Jumping Knowledge
#         if jk:
#             self.jk = JumpingKnowledge(mode=jk, channels=hidden_channels, num_layers=num_layers)
#             final_channels = hidden_channels * num_layers if jk == 'cat' else hidden_channels
#         else:
#             self.jk = None
#             final_channels = hidden_channels
#
#         self.dropout = Dropout(0.0)
#         self.lin = Linear(final_channels, out_channels)
#
#     def forward(self, x, edge_index, batch=None):
#         # Transform input features to hidden_channels
#         x = self.input_proj(x)
#
#         xs = []
#         for conv, norm in zip(self.convs, self.norms):
#             x = conv(x, edge_index)
#             x = norm(x)
#             x = self.act(x)
#             x = self.dropout(x)
#             xs.append(x)
#
#         if self.jk:
#             x = self.jk(xs)
#         else:
#             x = xs[-1]
#
#         x = global_max_pool(x, batch) if batch is not None else x
#         x = self.dropout(x)  # apply dropout last layer
#         x = self.lin(x)
#         return x



import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential, Linear, ReLU, Embedding, BatchNorm1d
from torch_geometric.nn import PNAConv, global_add_pool
from torch_geometric.typing import Adj
from typing import Dict, Any, Optional, List

class GraphGPS(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        pe_dim: int,
        num_layers: int,
        heads: int,
        dropout_rate: float,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
        edge_dim: Optional[int] = None,
        act: str = 'relu',
        jk: Optional[str] = None,
        deg: Optional[Tensor] = None,
        aggregators: List[str] = ['mean', 'max', 'std'],
        scalers: List[str] = ['identity'],
    ):
        super(GraphGPS, self).__init__()

        # Node and edge embeddings
        self.node_emb = Embedding(in_channels, hidden_channels - pe_dim)
        self.edge_emb = Embedding(edge_dim, hidden_channels) if edge_dim is not None else None

        # Positional encoding processing
        self.pe_lin = Linear(pe_dim, pe_dim)
        self.pe_norm = BatchNorm1d(pe_dim)

        # GPSConv layers
        self.convs = ModuleList()
        for _ in range(num_layers):
            # Local message-passing layer (PNAConv)
            conv = GPSConv(
                channels=hidden_channels,
                conv=PNAConv(
                    hidden_channels,
                    hidden_channels,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                ),
                heads=heads,
                dropout=dropout_rate,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs or {},
            )
            self.convs.append(conv)

        # Jumping Knowledge (JK) for combining layer outputs
        if jk:
            self.jk = JumpingKnowledge(mode=jk, channels=hidden_channels, num_layers=num_layers)
            final_channels = hidden_channels * num_layers if jk == 'cat' else hidden_channels
        else:
            self.jk = None
            final_channels = hidden_channels

        # MLP for downstream tasks
        self.mlp = Sequential(
            Linear(final_channels, final_channels // 2),
            ReLU() if act == 'relu' else torch.nn.Identity(),
            Linear(final_channels // 2, final_channels // 4),
            ReLU() if act == 'relu' else torch.nn.Identity(),
            Linear(final_channels // 4, out_channels),
        )

        # Dropout
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        # Process positional encodings
        x_pe = self.pe_norm(pe)
        x_pe = self.pe_lin(x_pe)

        # Combine node features and positional encodings
        x = torch.cat((self.node_emb(x.squeeze(-1)), x_pe), dim=-1)

        # Embed edge features (if provided)
        if self.edge_emb is not None and edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)

        # Apply GPSConv layers
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index, batch=batch, edge_attr=edge_attr)
            x = self.dropout(x)
            xs.append(x)

            # Combine outputs using Jumping Knowledge (if enabled)
            if self.jk:
                x = self.jk(xs)
            else:
                x = xs[-1]

        # Graph-level pooling
        x = global_max_pool(x, batch) if batch is not None else x

        # Final MLP for predictions
        return self.mlp(x)


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
            self.gnn = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm, aggr="max")
        elif model_type == 'GIN':
            self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type in ['GAT', 'GATv2']:
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm, heads=num_heads, v2='v2' in model_type)
        elif model_type in 'GatedGCN':
            self.gnn = GatedGCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm) # check attn type from paper
        elif model_type == 'PNA':
            self.gnn = PNA(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers,aggregators=['mean', 'max', 'std'],scalers=['identity'], dropout=0.0, act=act, norm=None, deg=deg, jk=None)
        elif model_type == 'GPS':
            self.gnn = GraphGPS(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=out_channels, heads=num_heads, dropout_rate=0.0, act=act, jk=None, deg=deg, pe_dim=20) # 'cat'

        # Final layer
        self.attention_pool = AttentionalAggregation(gate_nn=Linear(out_channels, 1))
        self.regressor = Linear(out_channels, params['out_features'])
        self.weight_layer = Linear(out_channels, 1)
        # self.classifier = Linear(out_channels, 2)  # Classifier layer, outputs two logits for binary classification
        # self.classifier = Linear(out_channels, 8)  # Classification head for jammer direction (8 cardinal directions)
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

        x, edge_index, edge_weight, wcl_pred = data.x, data.edge_index, data.edge_weight, data.wcl_pred

        # print("x[-1]: ", x[-1])
        # quit()

        # Handle based on the type of GNN
        if isinstance(self.gnn, GCN):
            # GCN supports edge weights
            x = self.gnn(x, edge_index, edge_weight=edge_weight)
        elif isinstance(self.gnn, GAT) or isinstance(self.gnn, GraphGPS):
            x = self.gnn(x, edge_index, edge_attr=edge_weight)
        else:
            # Fallback for other types if no specific handling is needed
            x = self.gnn(x, edge_index)

        # Extract the super node representation (assuming last node is the super node)
        super_node_feature = x[-1]
        # Remove the super node and apply pooling to get the graph representation
        graph_representation = self.pooling(x[:-1], data.batch)
        # Compute GNN-based prediction
        gnn_prediction = self.regressor(graph_representation)
        # Compute weight using super node feature
        weight = torch.sigmoid(self.weight_layer(super_node_feature))
        # Compute final prediction as a weighted sum of GNN and WCL results
        final_prediction = weight * gnn_prediction + (1 - weight) * wcl_pred
        return final_prediction


        # # Apply GNN layers
        # if params['pooling'] == 'max':
        #     x = global_max_pool(x, data.batch)
        # elif params['pooling'] == 'mean':
        #     x = global_mean_pool(x, data.batch)
        # elif params['pooling'] == 'sum':
        #     x = global_add_pool(x, data.batch)
        # elif params['pooling'] == 'att':
        #     x = self.attention_pool(x, data.batch)

        # x = self.dropout(x)  # apply dropout last layer
        #
        # # Regression and classification outputs
        # reg_output = self.regressor(x)
        #
        # return reg_output


def pooling(self, x, batch):
    """Applies the chosen pooling method."""
    if params['pooling'] == 'max':
        return global_max_pool(x, batch)
    elif params['pooling'] == 'mean':
        return global_mean_pool(x, batch)
    elif params['pooling'] == 'sum':
        return global_add_pool(x, batch)
    elif params['pooling'] == 'att':
        return self.attention_pool(x, batch)