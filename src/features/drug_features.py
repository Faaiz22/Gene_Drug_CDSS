import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np

class DrugFeaturizer:
    """
    Convert SMILES to molecular graphs and fingerprints.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.fp_radius = config['features']['ecfp_radius']
        self.fp_bits = config['features']['ecfp_bits']
    
    def smiles_to_graph(self, smiles: str) -> dgl.DGLGraph:
        """
        Convert SMILES to DGL graph.
        """
        mol = Chem.MolFromSmiles(smiles)
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetDegree(),  # Degree
                atom.GetTotalNumHs(),  # Hydrogens
                atom.GetImplicitValence(),  # Valence
                int(atom.GetIsAromatic()),  # Aromaticity
            ]
            atom_features.append(features)
        
        atom_features = torch.tensor(atom_features, dtype=torch.float32)
        
        # Edges (bonds)
        src, dst = [], []
        edge_features = []
        
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Add both directions (undirected graph)
            src.extend([i, j])
            dst.extend([j, i])
            
            bond_type = [
                float(bond.GetBondTypeAsDouble()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing())
            ]
            edge_features.extend([bond_type, bond_type])
        
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        
        # Create DGL graph
        g = dgl.graph((src, dst))
        g.ndata['feat'] = atom_features
        g.edata['feat'] = edge_features
        
        return g
    
    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Generate ECFP4 fingerprint.
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.fp_radius, 
            nBits=self.fp_bits
        )
        return np.array(fp)
    
    def compute_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Compute molecular descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
