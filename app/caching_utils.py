import streamlit as st
import yaml
import torch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel
import asyncio # <-- Import asyncio

# Import your actual project files
from src.utils.api_clients import DataEnricher
from src.models.dti_model import DTIModel
from src.molecular_3d.conformer_generator import smiles_to_3d_graph

# ... (ProteinFeaturizer class remains unchanged) ...
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
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding

# --- Caching Functions ---

@st.cache_resource
def load_config():
    """Loads the config file."""
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

@st.cache_resource
def load_enricher(_config):
    """Loads the DataEnricher."""
    return DataEnricher(_config)

@st.cache_resource
def load_sync_resources(_config, device):
    """Loads synchronous resources: DTI model and Protein featurizer."""
    print("--- Loading ML models... ---")
    
    # Load Protein Featurizer
    protein_model_path = _config['model'].get('protein_model_path', 'facebook/esm2_t6_8M_UR50D')
    protein_featurizer = ProteinFeaturizer(protein_model_path, device)

    # Load DTI Model
    dti_model = DTIModel(_config).to(device)
    weights_path = _config['paths']['model_weights']
    
    if torch.cuda.is_available():
        dti_model.load_state_dict(torch.load(weights_path))
    else:
        dti_model.load_state_dict(torch.load(weights_path, map_location=device))
    
    dti_model.eval()
    print("--- ML models loaded successfully. ---")
    return dti_model, protein_featurizer

@st.cache_data(max_entries=100)
async def featurize_pair(_enricher: DataEnricher, _protein_featurizer: ProteinFeaturizer, 
                         gene_id: str, chem_id: str, config: dict, device: torch.device):
    """
    Asynchronously fetches data and then synchronously featurizes it.
    """
    
    # 1. Fetch data concurrently
    try:
        smiles_task = _enricher.fetch_smiles(chem_id)
        sequence_task = _enricher.fetch_sequence(gene_id)
        
        smiles, sequence = await asyncio.gather(smiles_task, sequence_task)
        
    except Exception as e:
        raise RuntimeError(f"Failed during API data fetching: {e}")

    if not smiles:
        raise ValueError(f"Could not retrieve SMILES for Chemical ID: {chem_id}")
    if not sequence:
        raise ValueError(f"Could not retrieve sequence for Gene ID: {gene_id}")

    # 2. Featurize Drug (Sync)
    graph = smiles_to_3d_graph(smiles, config)
    if graph is None:
        raise ValueError("Failed to generate 3D graph for SMILES.")
    
    # 3. Featurize Protein (Sync)
    protein_emb = _protein_featurizer.featurize(sequence)
    
    # 4. Create a PyG batch
    graph_batch = DataLoader([graph], batch_size=1).to(device)
    protein_emb = protein_emb.to(device)

    return graph_batch, protein_emb, smiles, sequence
