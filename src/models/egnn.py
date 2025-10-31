"""
E(n)-Equivariant Graph Neural Network (EGNN)
Preserves rotational and translational symmetries in 3D space.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class EGNNLayer(MessagePassing):
    """
    Single E(n)-Equivariant Graph Neural Network Layer
    """
    def __init__(self, hidden_dim, edge_feat_dim=0):
        super(EGNNLayer, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Coordinate MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        Args:
            h: Node features [num_nodes, hidden_dim]
            pos: Node 3D coordinates [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim] (optional)
        
        Returns:
            h_out: Updated node features
            pos_out: Updated coordinates
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        
        # Message passing
        h_out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        
        # Update coordinates
        pos_out = pos + self.coord_update(h, pos, edge_index)
        
        # Update node features
        h_out = h + self.node_mlp(torch.cat([h, h_out], dim=-1))
        
        return h_out, pos_out
    
    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """
        Construct messages for each edge
        """
        # Compute distance
        rel_pos = pos_i - pos_j
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # Concatenate features
        if edge_attr is not None:
            msg_input = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([h_i, h_j, dist], dim=-1)
        
        # Compute message
        msg = self.message_mlp(msg_input)
        
        return msg
    
    def coord_update(self, h, pos, edge_index):
        """
        Update coordinates in an equivariant manner
        """
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # Compute coordinate weights
        coord_weights = self.coord_mlp(h[row])
        
        # Normalize relative positions
        rel_pos_normalized = rel_pos / (dist + 1e-8)
        
        # Weighted update
        coord_diff = coord_weights * rel_pos_normalized
        
        # Aggregate
        coord_update = torch.zeros_like(pos)
        coord_update.index_add_(0, col, coord_diff)
        
        return coord_update


class EGNN(nn.Module):
    """
    Multi-layer E(n)-Equivariant Graph Neural Network
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4, edge_feat_dim=0):
        super(EGNN, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        
        # Input projection
        self.node_embedding = nn.Linear(in_dim, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim) 
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, h, pos, edge_index, edge_attr=None, batch=None):
        """
        Args:
            h: Node features [num_nodes, in_dim]
            pos: Node 3D coordinates [num_nodes, 3]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feat_dim] (optional)
            batch: Batch assignment [num_nodes] (optional)
        
        Returns:
            node_features: Final node embeddings [num_nodes, out_dim]
            final_pos: Final coordinates [num_nodes, 3]
        """
        # Project input features
        h = self.node_embedding(h)
        
        # Apply EGNN layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
        
        # Output projection
        node_features = self.output_layer(h)
        
        return node_features, pos
