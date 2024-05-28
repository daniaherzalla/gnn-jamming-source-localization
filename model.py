import torch
from torch_geometric.nn import GATConv, global_mean_pool, AttentionalAggregation
from torch.nn import Linear, ReLU, Module
import torch.nn.functional as F
from utils import set_random_seeds

set_random_seeds()


class GraphAttentionNetwork(torch.nn.Module):
    """
    A Graph Attention Network (GAT) model for predicting jammer coordinates.

    Args:
        dropout_rate (float): The dropout rate for regularization.
        num_heads (int): The number of attention heads in the GAT layers.
    """
    def __init__(self, dropout_rate, num_heads):
        super(GraphAttentionNetwork, self).__init__()
        self.gat1 = GATConv(6, 32, heads=num_heads)  # Input feature dimension is 6, adjust as per your data
        self.gat2 = GATConv(32 * num_heads, 64, heads=num_heads)
        self.attention_pool = AttentionalAggregation(gate_nn=Linear(64 * num_heads, 1))
        self.regressor = torch.nn.Linear(64 * num_heads, 3)  # Assuming output of GlobalAttention is [64]
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, data):
        """
        Forward pass for the GraphAttentionNetwork.

        Args:
            data (Data): The input data containing node features and edge indices.

        Returns:
            Tensor: The predicted coordinates of the jammer.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        # x = self.dropout(x)
        x = F.relu(self.gat2(x, edge_index))
        # x = self.dropout(x)
        x = self.attention_pool(x, data.batch)  # Apply attention pooling to get a single vector for the graph
        # x = global_mean_pool(x, data.batch)  # Pooling to predict a single output per graph
        x = self.dropout(x)  # apply dropout last layer
        x = self.regressor(x)  # Predict the jammer's coordinates
        return x
