import yaml
import torch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModel


# Import your project's modules
from src.utils.api_clients import DataEnricher
from src.models.dti_model import DTIModel
from src.molecular_3d.conformer_generator import smiles_to_3d_graph

# ... (all other imports)
# REMOVE: import nest_asyncio
# REMOVE: from concurrent.futures import ThreadPoolExecutor

class CoreProcessor:
    def __init__(self, config):
        # ... (your existing __init__)
        pass

    # ... (all other methods like load_model, run_model, etc.)

    # REMOVED: def process_single_pair(self, gene_id, chem_id): ...
    # This function is GONE.

    async def process_pair_logic(self, gene_id: str, chem_id: str) -> dict:
        """
        Asynchronously processes a single gene-drug pair.
        This is the core logic for batch processing.
        """
        # REMOVED: nest_asyncio.apply()
        
        start_time = time.time()
        result = {"gene_id": gene_id, "chem_id": chem_id, "status": "Failed"}
        
        try:
            # 1. Validate inputs
            gene_id = validate_gene_id(gene_id)
            chem_id = validate_chem_id(chem_id)

            # 2. Fetch data concurrently
            async with self.get_enricher() as enricher:
                smiles_task = enricher.fetch_smiles(chem_id)
                sequence_task = enricher.fetch_sequence(gene_id)
                
                smiles, sequence = await asyncio.gather(
                    smiles_task, tran_task, return_exceptions=True
                )

            # 3. Handle data fetching errors
            if isinstance(smiles, Exception):
                raise smiles
            if isinstance(sequence, Exception):
                raise sequence
                
            result["smiles"] = smiles
            result["sequence"] = "..." # Don't store full sequence in result table

            # 4. Run model
            model_output = self.run_model(smiles, sequence)
            
            result.update({
                "status": "Success",
                "prediction": model_output["prediction"],
                "probability": model_output["probability"]
            })

        except (DataFetchException, ValidationException, FeaturizationException, ModelException) as e:
            result["error"] = e.message
        except Exception as e:
            result["error"] = f"An unexpected error: {str(e)}"
            
        result["runtime"] = time.time() - start_time
        return result

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
    # src/core_processing.py (ADDITIONS)

# Keep your existing async function, but add this:

def process_pair_logic_sync(
    gene_id: str, 
    chem_id: str, 
    enricher, 
    protein_featurizer, 
    model, 
    config: dict, 
    device: torch.device
) -> dict:
    """
    Synchronous wrapper for process_pair_logic.
    Handles event loop creation for Streamlit compatibility.
    """
    try:
        # Try to get existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running (Streamlit case), use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            result = loop.run_until_complete(
                process_pair_logic(
                    gene_id, chem_id, enricher, protein_featurizer, 
                    model, config, device
                )
            )
        else:
            result = loop.run_until_complete(
                process_pair_logic(
                    gene_id, chem_id, enricher, protein_featurizer, 
                    model, config, device
                )
            )
    except RuntimeError:
        # No event loop, create new one
        result = asyncio.run(
            process_pair_logic(
                gene_id, chem_id, enricher, protein_featurizer, 
                model, config, device
            )
        )
    
    return result


async def process_pair_logic(
    gene_id: str, 
    chem_id: str, 
    enricher, 
    protein_featurizer, 
    model, 
    config: dict, 
    device: torch.device
) -> dict:
    """
    The core logic for a single prediction, from IDs to a result.
    This is async and returns a dictionary.
    """
    
    # 1. Fetch data concurrently
    try:
        smiles, sequence = await asyncio.gather(
            enricher.fetch_smiles(chem_id),
            enricher.fetch_sequence(gene_id)
        )
    except Exception as e:
        raise RuntimeError(f"Failed during API data fetching for ({gene_id}, {chem_id}): {e}")

    if not smiles:
        raise ValueError(f"Could not retrieve SMILES for: {chem_id}")
    if not sequence:
        raise ValueError(f"Could not retrieve sequence for: {gene_id}")

    # 2. Featurize Drug (Sync - move to thread if slow)
    from src.molecular_3d.conformer_generator import smiles_to_3d_graph
    graph = await asyncio.to_thread(smiles_to_3d_graph, smiles, config)
    
    if graph is None:
        raise ValueError(f"Failed to generate 3D graph for SMILES: {smiles}")
    
    # 3. Featurize Protein (Sync - move to thread)
    protein_emb = await asyncio.to_thread(protein_featurizer.featurize, sequence)
    
    # 4. Create a PyG batch
    from torch_geometric.data import Batch
    graph_batch = Batch.from_data_list([graph]).to(device)
    protein_emb_batch = protein_emb.unsqueeze(0).to(device)

    # 5. Run prediction (Sync - move to thread)
    def run_prediction():
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(graph_batch, protein_emb_batch)
            probability = torch.sigmoid(logits).item()
        return probability
    
    probability = await asyncio.to_thread(run_prediction)

    return {
        "gene_id": gene_id,
        "chem_id": chem_id,
        "smiles": smiles,
        "sequence_length": len(sequence),
        "probability": probability,
        "graph_nodes": graph_batch.x.shape[0],
        "protein_embedding_shape": list(protein_emb_batch.shape)
    }
