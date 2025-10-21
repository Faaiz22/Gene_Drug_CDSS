from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


def generate_3d_conformer(smiles: str, conformer_optimize_steps: int) -> Optional[Chem.Mol]:
    """Generate optimized 3D conformer from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    success = AllChem.EmbedMolecule(mol, params) == 0
    if not success:
        return None

    AllChem.UFFOptimizeMolecule(mol, maxIters=conformer_optimize_steps)
    return mol


def smiles_to_3d_graph(smiles: str, config: dict) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric graph with 3D coordinates"""
    mol = generate_3d_conformer(smiles, config['processing']['conformer_optimize_steps'])
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge()
        ])
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_features = [
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing())
        ]
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bond_features, bond_features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    # 3D coordinates
    conf = mol.GetConformer()
    pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
