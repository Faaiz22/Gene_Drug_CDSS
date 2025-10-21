import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    """E(n)-Equivariant Graph Neural Network Layer"""

    def __init__(self, node_dim, edge_dim, output_dim):
        super().__init__(aggr='mean')
        self.lin_node = nn.Linear(node_dim, output_dim)
        self.lin_edge = nn.Linear(edge_dim, output_dim)
        self.lin_message = nn.Linear(2 * output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, edge_attr, pos):
        x_in = self.lin_node(x)
        e_in = self.lin_edge(edge_attr)
        return self.propagate(edge_index, x=x_in, edge_attr=e_in, pos=pos)

    def message(self, x_i, x_j, edge_attr, pos_i, pos_j):
        d = torch.norm(pos_i - pos_j, dim=1, keepdim=True)
        msg = torch.cat([x_j, edge_attr], dim=1)
        msg = self.lin_message(msg)
        msg = msg * torch.exp(-d)
        return self.norm(msg)
