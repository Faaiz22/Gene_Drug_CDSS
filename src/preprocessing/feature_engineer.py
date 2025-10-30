"""
Feature Engineer for 3D Drug-Protein Interaction

This module performs the state-of-the-art featurization:
1.  Drug: Converts SMILES to a 3D graph with atomic coordinates (for EGNN).
2.  Protein: Converts amino acid sequence to a high-dimensional embedding
    using the ESM-2 Protein Language Model.
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import EsmModel, EsmTokenizer
import logging

# Set up logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        """
        Initializes the featurizer, loading the ESM-2 model.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ESM-2 Model and Tokenizer
        # Using a smaller ESM-2 model for faster inference.
        # For SOTA results, 'facebook/esm2_t33_650M_UR50D' is common.
        esm_model_name = self.config.get('esm_model', 'facebook/esm2_t12_35M_UR50D')
        
        logger.info(f"Loading ESM-2 model: {esm_model_name}...")
        try:
            self.esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
            self.esm_model = EsmModel.from_pretrained(esm_model_name).to(self.device)
            self.esm_model.eval()  # Set to evaluation mode
            logger.info("ESM-2 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ESM-2 model: {e}")
            raise

    def _get_drug_graph(self, smiles_string: str) -> Data:
        """
        Converts a SMILES string into a 3D graph (Data object).
        Generates 3D conformer and extracts atomic features and coordinates.
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles_string}")

        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status == -1:
            logger.warning(f"Could not generate 3D conformer for {smiles_string}. Trying with random coords.")
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(useRandomCoords=True))
            if status == -1:
                 raise ValueError(f"Failed to generate 3D conformer even with random coords for {smiles_string}")
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            logger.warning(f"MMFF optimization failed for {smiles_string}: {e}. Using unoptimized conformer.")

        conformer = mol.GetConformer()
        positions = torch.tensor(conformer.GetPositions(), dtype=torch.float)

        # Extract atomic features (e.g., atom type)
        # This can be expanded (atomic_num, charge, is_aromatic, etc.)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom.GetAtomicNum())
        
        x = torch.tensor(atom_features, dtype=torch.long).view(-1, 1)

        # Extract edge index (bonds)
        edge_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.extend([[i, j], [j, i]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create the PyG Data object
        data = Data(x=x, edge_index=edge_index, pos=positions)
        return data

    @torch.no_grad()
    def _get_protein_embedding(self, sequence: str) -> torch.Tensor:
        """
        Converts an amino acid sequence into a fixed-size embedding
        using the ESM-2 model.
        """
        # Truncate sequence if it's too long for the model (e.g., >1022 tokens)
        if len(sequence) > 1022:
            sequence = sequence[:1022]
            
        tokens = self.esm_tokenizer(sequence, return_tensors='pt', add_special_tokens=True).to(self.device)
        output = self.esm_model(**tokens)
        
        # Use mean pooling of the last hidden state to get a single vector
        # for the entire protein.
        # We ignore <cls> and <eos> tokens (indices 0 and -1)
        embedding = output.last_hidden_state[0, 1:-1].mean(dim=0)
        return embedding.cpu()

    def featurize(self, smiles: str, sequence: str) -> Data:
        """
        Main featurization function.
        Combines the 3D drug graph and the ESM-2 protein embedding
        into a single Data object.
        """
        # 1. Get 3D drug graph
        drug_data = self._get_drug_graph(smiles)
        
        # 2. Get ESM-2 protein embedding
        protein_embedding = self._get_protein_embedding(sequence)
        
        # 3. Store the protein embedding inside the drug's Data object
        # This is a common PyG trick to keep pairs together in a batch
        drug_data.protein_embedding = protein_embedding.unsqueeze(0) # Add batch dim
        
        return drug_data
