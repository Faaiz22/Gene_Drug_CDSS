import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv

class DrugGNN(nn.Module):
    """
    Graph Neural Network encoder for drug molecules.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.node_feat_dim = config['model']['drug_node_dim']
        self.edge_feat_dim = config['model']['drug_edge_dim']
        self.hidden_dim = config['model']['drug_hidden_dim']
        self.output_dim = config['model']['drug_output_dim']
        self.num_layers = config['model']['drug_num_layers']
        
        # Input projection
        self.node_encoder = nn.Linear(self.node_feat_dim, self.hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                GATConv(
                    in_feats=self.hidden_dim,
                    out_feats=self.hidden_dim,
                    num_heads=4,
                    feat_drop=0.2,
                    attn_drop=0.2,
                    activation=nn.ReLU()
                )
            )
        
        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            g: Batched DGL graph
            
        Returns:
            Graph-level embeddings [batch_size, output_dim]
        """
        h = self.node_encoder(g.ndata['feat'])
        
        # Message passing
        for conv in self.convs:
            h = conv(g, h).flatten(1)  # Flatten multi-head outputs
        
        # Global pooling
        with g.local_scope():
            g.ndata['h'] = h
            # Mean + Max + Sum pooling
            h_mean = dgl.mean_nodes(g, 'h')
            h_max = dgl.max_nodes(g, 'h')
            h_sum = dgl.sum_nodes(g, 'h')
            h_graph = torch.cat([h_mean, h_max, h_sum], dim=1)
        
        return self.readout(h_graph)
