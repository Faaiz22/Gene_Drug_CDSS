import torch
import torch.nn as nn

class InteractionPredictor(nn.Module):
    """
    Complete gene-drug interaction prediction model.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.drug_encoder = DrugGNN(config)
        self.protein_encoder = ProteinCNN(config)
        
        drug_dim = config['model']['drug_output_dim']
        protein_dim = config['model']['protein_output_dim']
        fusion_dim = drug_dim + protein_dim
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, drug_graph: dgl.DGLGraph, protein_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict interaction probability.
        
        Returns:
            Logits [batch_size, 1]
        """
        drug_emb = self.drug_encoder(drug_graph)
        protein_emb = self.protein_encoder(protein_seq)
        
        # Concatenate and predict
        combined = torch.cat([drug_emb, protein_emb], dim=1)
        logits = self.fusion(combined)
        
        return logits
