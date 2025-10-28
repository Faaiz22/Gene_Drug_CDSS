import logging
from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from ..utils.exceptions import FeaturizationException

# Set up a logger for this module
log = logging.getLogger(__name__)

def generate_3d_conformer(smiles: str, conformer_optimize_steps: int) -> Optional[Chem.Mol]:
    """
    Generate optimized 3D conformer from SMILES.
    Returns None if generation fails at any step.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(f"RDKit failed to parse SMILES: {smiles}")
            return None
    except Exception as e:
        log.error(f"Error parsing SMILES {smiles}: {e}")
        return None

    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        log.warning(f"Failed to add Hydrogens to SMILES {smiles}: {e}")
        return None # Fails for some complex structures

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True # Add this for robustness

    embed_success = -1
    try:
        embed_success = AllChem.EmbedMolecule(mol, params)
    except Exception as e:
        log.warning(f"RDKit EmbedMolecule failed for SMILES {smiles}: {e}")
        return None

    if embed_success != 0:
        log.warning(f"Failed to embed molecule for SMILES: {smiles} (Error code: {embed_success})")
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=conformer_optimize_steps)
    except Exception as e:
        log.warning(f"RDKit UFFOptimizeMolecule failed for SMILES {smiles}: {e}")
        # We can choose to return the unoptimized molecule, but for E(3) models,
        # a bad conformation is bad data. Failing fast is better.
        return None
        
    return mol


def smiles_to_3d_graph(smiles: str, config: dict) -> Optional[Data]:
    """
    Convert SMILES to PyTorch Geometric graph with 3D coordinates.
    Raises FeaturizationException on failure.
    """
    mol = generate_3d_conformer(smiles, config['processing']['conformer_optimize_steps'])
    if mol is None:
        # This is the exception that the UI can catch and show to the user.
        raise FeaturizationException(
            "Failed to generate 3D conformer.",
            f"Could not generate a valid 3D structure for SMILES: {smiles[:30]}..."
        )

    try:
        n_atoms = mol.GetNumAtoms()

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                float(atom.GetIsAromatic()), # Use float for consistency
                atom.GetFormalCharge()
            ])
        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge features
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = [
                bond.GetBondTypeAsDouble(),
                float(bond.GetIsAromatic()),
                float(bond.GetIsInRing())
            ]
            # Add bonds in both directions
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([bond_features, bond_features])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # Handle case with no bonds (e.g., single atom [He])
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, config['model']['drug_edge_dim']), dtype=torch.float) # Use config dim

        # 3D coordinates
        conf = mol.GetConformer() # We know this exists from generate_3d_conformer
        pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    except Exception as e:
        log.error(f"Failed to featurize graph from RDKit Mol object for {smiles}: {e}")
        raise FeaturizationException(
            "Failed to convert 3D molecule to graph.",
            f"Error during graph featurization: {e}"
        )
