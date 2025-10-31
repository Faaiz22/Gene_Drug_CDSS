"""
Feature Engineering for Drug-Target Interaction Prediction
Handles 3D molecular conformer generation and protein sequence embedding
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from transformers import EsmTokenizer, EsmModel


class FeatureEngineer:
    """
    Featurizes drug SMILES and protein sequences for DTI prediction
    """
    
    def __init__(self, config=None, device=None):
        """
        Initialize FeatureEngineer
        
        Args:
            config: Configuration dictionary (optional)
            device: torch device (optional, defaults to 'cpu')
        """
        self.config = config or {}
        self.device = device or torch.device('cpu')
        
        # Load ESM-2 protein language model
        print("Loading ESM-2 model...")
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.protein_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.protein_model.to(self.device)
        self.protein_model.eval()
        print("ESM-2 model loaded successfully.")
        
    def smiles_to_graph(self, smiles):
        """
        Convert SMILES to 3D molecular graph
        
        Args:
            smiles: SMILES string
            
        Returns:
            node_features: [num_atoms, 9]
            edge_index: [2, num_bonds]
            pos: [num_atoms, 3]
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            raise ValueError(f"Failed to generate 3D conformer: {e}")
        
        # Get conformer
        conf = mol.GetConformer()
        
        # Extract atom features
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                atom.GetIsAromatic(),
                atom.GetMass(),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons(),
                atom.IsInRing()
            ]
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Extract 3D coordinates
        pos = []
        for i in range(mol.GetNumAtoms()):
            position = conf.GetAtomPosition(i)
            pos.append([position.x, position.y, position.z])
        
        pos = torch.tensor(pos, dtype=torch.float)
        
        # Extract edges
        edge_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.append([i, j])
            edge_list.append([j, i])  # Undirected graph
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return node_features, edge_index, pos
    
    def embed_protein(self, sequence):
        """
        Generate protein sequence embedding using ESM-2
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            embedding: [protein_embed_dim] tensor
        """
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.protein_model(**inputs)
            # Use mean of sequence tokens (excluding special tokens)
            embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze(0)
        
        return embedding
    
    def featurize(self, smiles, sequence, label):
        """
        Featurize a drug-target pair
        
        Args:
            smiles: Drug SMILES string
            sequence: Protein sequence string
            label: Interaction label (0 or 1)
            
        Returns:
            data: torch_geometric.data.Data object
        """
        try:
            # Process drug
            node_features, edge_index, pos = self.smiles_to_graph(smiles)
            
            # Process protein
            protein_embed = self.embed_protein(sequence)
            
            # Create PyG Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                pos=pos,
                protein_embed=protein_embed.unsqueeze(0),  # Add batch dimension
                y=torch.tensor([label], dtype=torch.float)
            )
            
            return data
            
        except Exception as e:
            # Return None for failed featurizations
            print(f"Featurization failed: {e}")
            return None
