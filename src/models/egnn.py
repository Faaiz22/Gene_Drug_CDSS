"""
DTI Prediction Model: EGNN + ESM-2 + Cross-Attention

This model implements the state-of-the-art architecture:
1.  EGNN: Encodes the 3D drug conformer graph.
2.  Protein Encoder: A simple MLP to project the ESM-2 embedding.
3.  Cross-Attention: Fuses the 3D drug atom embeddings with the
    protein embedding.
4.  Prediction Head: Predicts the final interaction probability.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# Import the EGNN implementation from your repository
try:
    from .egnn import EGNN
except ImportError:
    raise ImportError("Failed to import EGNN from .egnn. Make sure src/models/egnn.py exists.")

class DTIPredictor(nn.Module):
    def __init__(self, 
                 protein_emb_dim: int, 
                 drug_atom_feature_dim: int = 1, # Input dim for atoms (e.g., atomic num)
                 egnn_hidden_dim: int = 128,
                 egnn_layers: int = 4,
                 cross_attn_heads: int = 4,
                 dropout_rate: float = 0.1):
        """
        Initializes the model layers.
        
        Args:
            protein_emb_dim: Dimension of the ESM-2 embedding (e.g., 320 for the 35M model).
            drug_atom_feature_dim: Number of input features for each atom.
            egnn_hidden_dim: Internal dimension for the EGNN and attention layers.
            egnn_layers: Number of EGNN message passing layers.
            cross_attn_heads: Number of heads for the cross-attention layer.
            dropout_rate: Dropout for regularization.
        """
        super().__init__()
        
        # --- Drug Atom Embedding ---
        # Convert atomic numbers (long) into dense vectors (float)
        # We'll use a 100-dim embedding for atomic numbers 1-99
        self.atom_embedding = nn.Embedding(100, egnn_hidden_dim)

        # --- 3D Drug Encoder (EGNN) ---
        # Takes in atomic embeddings (h) and positions (x)
        self.egnn = EGNN(
            in_node_nf=egnn_hidden_dim,
            hidden_nf=egnn_hidden_dim,
            out_node_nf=egnn_hidden_dim,
            n_layers=egnn_layers,
            attention=True,
            normalize=True,
            residual=True
        )

        # --- Protein Embedding Encoder ---
        # Project ESM-2 embedding to the same hidden dimension
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_emb_dim, egnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # --- Drug-Protein Interaction (Cross-Attention) ---
        # The protein (context) will "attend" to each atom in the drug
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=egnn_hidden_dim,
            num_heads=cross_attn_heads,
            dropout=dropout_rate,
            batch_first=True # Expects (batch, seq, features)
        )

        # --- Final Prediction Head ---
        # Takes the fused representation and predicts a single logit
        self.prediction_head = nn.Sequential(
            nn.Linear(egnn_hidden_dim * 2, egnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(egnn_hidden_dim, 1) # Output a single logit for BCEWithLogitsLoss
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            data (torch_geometric.data.Batch): A batch of Data objects from
                                               our FeatureEngineer.
        
        Returns:
            torch.Tensor: A tensor of (batch_size, 1) logits.
        """
        
        # --- 1. Process Drug ---
        data.x shape: (num_atoms_in_batch, 1) - Atomic numbers
        data.pos shape: (num_atoms_in_batch, 3) - XYZ coordinates
        data.edge_index shape: (2, num_bonds_in_batch)
        data.batch shape: (num_atoms_in_batch) - Maps atoms to graphs
        
        # Get initial atom embeddings from atomic numbers
        atom_embeds = self.atom_embedding(data.x.squeeze()) # (num_atoms, hidden_dim)

        # Pass atom embeddings (h) and positions (x) through EGNN
        h_out shape: (num_atoms, hidden_dim)
        h_out, _ = self.egnn(
            h=atom_embeds, 
            x=data.pos, 
            edges=data.edge_index, 
            node_mask=None, # Assuming all nodes are real
            edge_mask=None
        )

        # --- 2. Process Protein ---
        data.protein_embedding shape: (batch_size, protein_emb_dim)
        protein_embed = self.protein_projection(data.protein_embedding)
        protein_embed shape: (batch_size, hidden_dim)
        
        # --- 3. Fuse with Cross-Attention ---
        # We want the protein to be the "context" for the drug atoms.
        
        # We need to map the batch-level protein embedding to the atom-level.
        # We "broadcast" the protein embedding to each atom in its respective graph.
        protein_embed_per_atom shape: (num_atoms, hidden_dim)
        protein_embed_per_atom = protein_embed[data.batch] 

        # Attention!
        # Query: Drug atom embeddings
        # Key: Protein context
        # Value: Protein context
        # We use unsqueeze(1) to add a "sequence length" dimension of 1
        
        # attn_input shape: (num_atoms, 1, hidden_dim)
        attn_input = h_out.unsqueeze(1)
        
        # context shape: (num_atoms, 1, hidden_dim)
        context = protein_embed_per_atom.unsqueeze(1)

        # attn_out shape: (num_atoms, 1, hidden_dim)
        attn_out, _ = self.cross_attention(
            query=attn_input,
            key=context,
            value=context
        )
        
        # Fused atom representations
        # We add the attention output to the original atom embedding (residual)
        fused_atom_embeds = h_out + attn_out.squeeze(1) # (num_atoms, hidden_dim)

        # --- 4. Pool and Predict ---
        
        # Pool all atom embeddings in each graph to get a single graph vector
        drug_vector = global_mean_pool(fused_atom_embeds, data.batch) # (batch_size, hidden_dim)
        
        # Concatenate the final drug vector and the protein vector
        combined_vector = torch.cat([drug_vector, protein_embed], dim=1) # (batch_size, hidden_dim * 2)
        
        # Get final prediction
        logit = self.prediction_head(combined_vector) # (batch_size, 1)
        
        return logit
