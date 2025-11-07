import torch
import torch.nn as nn
from typing import Dict, List

class ProteinFeaturizer:
    """
    Convert protein sequences to numerical features.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_length = config['features']['max_protein_length']
        
        # Amino acid alphabet
        self.aa_dict = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_dict['X'] = len(self.aa_dict)  # Unknown
        self.vocab_size = len(self.aa_dict)
    
    def sequence_to_indices(self, sequence: str) -> torch.Tensor:
        """
        Convert sequence to integer indices.
        """
        # Truncate or pad
        sequence = sequence[:self.max_length]
        indices = [self.aa_dict.get(aa, self.aa_dict['X']) for aa in sequence]
        
        # Pad to max_length
        if len(indices) < self.max_length:
            indices += [self.vocab_size] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def compute_composition(self, sequence: str) -> torch.Tensor:
        """
        Compute amino acid composition features.
        """
        composition = torch.zeros(self.vocab_size)
        for aa in sequence:
            idx = self.aa_dict.get(aa, self.aa_dict['X'])
            composition[idx] += 1
        
        # Normalize
        composition = composition / len(sequence)
        return composition
    
    def compute_physicochemical(self, sequence: str) -> torch.Tensor:
        """
        Compute physicochemical properties.
        """
        # Simple averages (can be extended)
        hydrophobicity_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        avg_hydrophobicity = np.mean([
            hydrophobicity_scale.get(aa, 0) for aa in sequence
        ])
        
        return torch.tensor([
            avg_hydrophobicity,
            len(sequence),
            sequence.count('C') / len(sequence),  # Cysteine fraction
        ], dtype=torch.float32)
