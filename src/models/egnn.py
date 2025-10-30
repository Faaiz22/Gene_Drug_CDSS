"""
E(n)-Equivariant Graph Neural Network (EGNN) Layer

This module implements a state-of-the-art GNN for 3D molecular processing,
respecting rotational and translational equivariance of 3D structures.
Reference: https://arxiv.org/abs/2102.09844
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, SiLU, Linear
from torch import Tensor
from typing import Optional, Tuple

class EGNN(MessagePassing):
    """
    E(n)-Equivariant Graph Neural Network Layer.

    Updates node features (h) and coordinates (x) in an equivariant manner.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_attr_dim: int = 0,
        aggr: str = 'add',
        norm_feat: bool = True,
        norm_coord: bool = False,
        act: nn.Module = SiLU()
    ):
        """
        Args:
            in_dim: Input node feature dimension.
            hidden_dim: Hidden layer dimension.
            out_dim: Output node feature dimension.
            edge_attr_dim: Edge feature dimension.
            aggr: Aggregation (e.g., 'add', 'mean').
            norm_feat: Apply layer norm to node features.
            norm_coord: Normalize coordinate updates.
            act: Activation function.
        """
        super().__init__(aggr=aggr)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.norm_feat = norm_feat
        self.norm_coord = norm_coord

        # Message network φ_e: [h_i, h_j, ||x_i-x_j||^2, a_ij] → msg
        phi_e_in = in_dim * 2 + 1 + edge_attr_dim
        self.phi_e = Sequential(
            Linear(phi_e_in, hidden_dim),
            act,
            Linear(hidden_dim, hidden_dim),
            act
        )
        # Node update network φ_h: [h, m] → h'
        self.phi_h = Sequential(
            Linear(in_dim + hidden_dim, hidden_dim),
            act,
            Linear(hidden_dim, out_dim)
        )
        # Coordinate update network φ_x: m → scalar
        self.phi_x = Sequential(
            Linear(hidden_dim, hidden_dim),
            act,
            Linear(hidden_dim, 1, bias=False)
        )
        # Optional LayerNorm
        if self.norm_feat:
            self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, 
        h: Tensor,         # [num_nodes, in_dim]
        x: Tensor,         # [num_nodes, 3]
        edge_index: Tensor, # [2, num_edges]
        edge_attr: Optional[Tensor] = None # [num_edges, edge_attr_dim]
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for EGNN.

        Returns:
            h_new: Updated node features.
            x_new: Updated node coordinates.
        """
        row, col = edge_index
        dist = x[row] - x[col]
        dist_sq = torch.sum(dist ** 2, dim=-1, keepdim=True) # [num_edges, 1]

        # Message passing
        m_i, x_new = self.propagate(
            edge_index, 
            h=h, 
            x=x, 
            dist_sq=dist_sq, 
            edge_attr=edge_attr
        )
        h_new = self.phi_h(torch.cat([h, m_i], dim=-1))
        if self.norm_feat:
            h_new = self.norm(h_new)
        return h_new, x_new

    def message(
        self, 
        h_i: Tensor, 
        h_j: Tensor, 
        dist_sq: Tensor, 
        edge_attr: Optional[Tensor]
    ) -> Tensor:
        """
        Calculates message m_ij via φ_e.
        """
        parts = [h_i, h_j, dist_sq]
        if edge_attr is not None:
            parts.append(edge_attr)
        msg_in = torch.cat(parts, dim=-1)
        m_ij = self.phi_e(msg_in)
        return m_ij

    def update(
        self, 
        m_i: Tensor,
        x: Tensor, 
        x_i: Tensor, 
        edge_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Coordinates update via φ_x and aggregation.
        """
        row, col = edge_index
        dist = x[row] - x[col]
        coord_scalar = self.phi_x(m_i) # [num_edges, 1]
        coord_update = dist * coord_scalar # [num_edges, 3]

        # Aggregate updates for each destination node (col)
        aggr_coord_update = torch.zeros_like(x)
        # Scatter updates (dim 0 = node index)
        aggr_coord_update = aggr_coord_update.scatter_add(
            0, col.unsqueeze(-1).expand_as(coord_update), coord_update
        )
        # Normalize if requested
        if self.norm_coord:
            # Normalization across each node's vector
            aggr_coord_update = (
                aggr_coord_update / (torch.norm(aggr_coord_update, dim=-1, keepdim=True) + 1e-8)
            )
        # Update coordinates
        x_new = x + aggr_coord_update
        return m_i, x_new

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(in_dim={self.in_dim}, '
            f'hidden_dim={self.hidden_dim}, out_dim={self.out_dim}, '
            f'edge_attr_dim={self.edge_attr_dim})'
        )

