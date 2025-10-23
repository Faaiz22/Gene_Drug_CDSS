import yaml
import torch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel
import asyncio

# Import your project's modules
from src.utils.api_clients import DataEnricher
from src.models.dti_model import DTIModel
from src.molecular_3d.conformer_generator import smiles_to_3d_graph

# --- Device & Config ---

def get_device() -> torch.device:
    """Gets the available torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config() -> dict:
    """Loads the main config file."""
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Resource Loaders (Headless) ---

class ProteinFeaturizer:
    """Loads and runs the protein language model."""
    def __init__(self, model_name_or_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def featurize(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

def load_protein_featurizer(config: dict, device: torch.device) -> ProteinFeaturizer:
    """Initializes the ProteinFeaturizer."""
    print("--- Loading Protein Language Model... ---")
    model_path = config['model'].get('protein_model_path', 'facebook/esm2_t6_8M_UR50D')
    return ProteinFeaturizer(model_path, device)

def load_dti_model(config: dict, device: torch.device) -> DTIModel:
    """Loads the trained DTI model."""
    print("--- Loading DTI Model... ---")
    model = DTIModel(config).to(device)
    weights_path = config['paths']['model_weights']
    
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    model.eval()
    return model

def load_enricher(config: dict) -> DataEnricher:
    """Loads the DataEnricher API client."""
    print("--- Loading DataEnricher... ---")
    return DataEnricher(config)

# --- Core Logic Function ---

async def process_pair_logic(gene_id: str, chem_id: str, 
                             enricher: DataEnricher, 
                             protein_featurizer: ProteinFeaturizer, 
                             model: DTIModel, 
                             config: dict, 
                             device: torch.device) -> dict:
    """
    The core logic for a single prediction, from IDs to a result.
    This is now async and returns a dictionary.
    """
    
    # 1. Fetch data concurrently
    try:
        smiles_task = enricher.fetch_smiles(chem_id)
        sequence_task = enricher.fetch_sequence(gene_id)
        smiles, sequence = await asyncio.gather(smiles_task, sequence_task)
    except Exception as e:
        raise RuntimeError(f"Failed during API data fetching for ({gene_id}, {chem_id}): {e}")

    if not smiles:
        raise ValueError(f"Could not retrieve SMILES for: {chem_id}")
    if not sequence:
        raise ValueError(f"Could not retrieve sequence for: {gene_id}")

    # 2. Featurize Drug (Sync)
    graph = smiles_to_3d_graph(smiles, config)
    if graph is None:
        raise ValueError(f"Failed to generate 3D graph for SMILES: {smiles}")
    
    # 3. Featurize Protein (Sync)
    protein_emb = protein_featurizer.featurize(sequence)
    
    # 4. Create a PyG batch
    graph_batch = DataLoader([graph], batch_size=1).to(device)
    protein_emb = protein_emb.to(device)

    # 5. Run prediction
    with torch.no_grad():
        logits, attn_weights = model(graph_batch, protein_emb)
        probability = torch.sigmoid(logits).item()

    return {
        "gene_id": gene_id,
        "chem_id": chem_id,
        "smiles": smiles,
        "sequence_length": len(sequence),
        "probability": probability,
        "graph_nodes": graph_batch.x.shape[0],
        "protein_embedding_shape": list(protein_emb.shape)
    }
