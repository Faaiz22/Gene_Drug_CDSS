import torch
import torch.nn as nn

class ProteinCNN(nn.Module):
    """
    1D CNN encoder for protein sequences.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.vocab_size = config['features']['protein_vocab_size']
        self.embedding_dim = config['model']['protein_embedding_dim']
        self.hidden_dim = config['model']['protein_hidden_dim']
        self.output_dim = config['model']['protein_output_dim']
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 1,  # +1 for padding
            embedding_dim=self.embedding_dim,
            padding_idx=self.vocab_size
        )
        
        # Convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(self.embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.embedding_dim, self.hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(self.embedding_dim, self.hidden_dim, kernel_size=7, padding=3)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(self.hidden_dim * 3)
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Sequence indices [batch_size, seq_len]
            
        Returns:
            Sequence embeddings [batch_size, output_dim]
        """
        # Embed
        x = self.embedding(x)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        
        # Multi-scale convolutions
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(x))
        h3 = torch.relu(self.conv3(x))
        
        # Global max pooling
        h1 = torch.max(h1, dim=2)[0]
        h2 = torch.max(h2, dim=2)[0]
        h3 = torch.max(h3, dim=2)[0]
        
        # Concatenate
        h = torch.cat([h1, h2, h3], dim=1)
        h = self.bn(h)
        
        return self.fc(h)
