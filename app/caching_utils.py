# app/caching_utils.py
import streamlit as st
import yaml
import torch
from torch_geometric.loader import DataLoader # To handle batching for the model
from transformers import AutoTokenizer, AutoModel

# Import your actual project files
from src.utils.api_clients import DataEnricher
from src.models.dti_model import DTIModel
from src.molecular_3d.conformer_generator import smiles_to_3d_graph

# --- Protein Featurizer (Inferred from your project) ---
# This class lives in 'src/preprocessing/feature_engineer.py'
# We define it here to show how to load it.
class ProteinFeaturizer:
    def __init__(self, model_name_or_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def featurize(self, sequence: str) -> torch.Tensor:
        """Converts a protein sequence to a single embedding vector."""
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs)
        # Get last hidden state and mean pool
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding

# --- Main Caching Function ---
@st.cache_resource
def load_all_resources():
    """
    Loads config, model, enricher, and featurizers at app start.
    """
    print("--- Loading all resources... ---")
    
    # 0. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load DataEnricher (for API calls)
    enricher = DataEnricher(config)

    # 3. Load Protein Featurizer (Transformer)
    # Assumes your config has: model: protein_model_path: "path/to/esm_or_protbert"
    protein_model_path = config['model'].get('protein_model_path', 'facebook/esm2_t6_8M_UR50D')
    protein_featurizer = ProteinFeaturizer(protein_model_path, device)

    # 4. Load DTI Model
    dti_model = DTIModel(config).to(device)
    weights_path = config['paths']['model_weights']
    
    if torch.cuda.is_available():
        dti_model.load_state_dict(torch.load(weights_path))
    else:
        dti_model.load_state_dict(torch.load(weights_path, map_location=device))
    
    dti_model.eval()
    print("--- All resources loaded successfully. ---")

    return config, enricher, protein_featurizer, dti_model, device

# --- Featurization Caching Function ---
@st.cache_data(max_entries=100) # Cache the 100 most recent computations
def featurize_pair(smiles: str, sequence: str, _protein_featurizer, config: dict, device: torch.device):
    """
    Caches the result of featurizing a single pair.
    _protein_featurizer is passed to ensure cache invalidation if it changes.
    """
    
    # 1. Featurize Drug
    graph = smiles_to_3d_graph(smiles, config) #
    if graph is None:
        raise ValueError("Failed to generate 3D graph for SMILES.")
    
    # 2. Featurize Protein
    protein_emb = _protein_featurizer.featurize(sequence)
    
    # 3. Create a PyG batch (even for a single item)
    graph_batch = DataLoader([graph], batch_size=1).to(device)
    protein_emb = protein_emb.to(device)

    return graph_batch, protein_emb
