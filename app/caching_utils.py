import streamlit as st
import torch

# Import all core logic functions
from src.core_processing import (
    get_device,
    load_config,
    load_protein_featurizer,
    load_dti_model,
    load_enricher,
    process_pair_logic
)

# --- Cached Resource Loaders ---

@st.cache_resource
def get_cached_device() -> torch.device:
    """Caches the torch device."""
    return get_device()

@st.cache_resource
def get_cached_config() -> dict:
    """Caches the app config."""
    return load_config()

@st.cache_resource
def get_cached_enricher() -> DataEnricher:
    """Caches the DataEnricher API client."""
    return load_enricher(get_cached_config())

@st.cache_resource
def get_cached_dti_model() -> DTIModel:
    """Caches the trained DTI model."""
    config = get_cached_config()
    device = get_cached_device()
    return load_dti_model(config, device)

@st.cache_resource
def get_cached_protein_featurizer() -> "ProteinFeaturizer":
    """Caches the protein language model."""
    config = get_cached_config()
    device = get_cached_device()
    return load_protein_featurizer(config, device)

# --- Cached Data/Prediction Function ---

@st.cache_data(max_entries=100, ttl="6h")
async def get_cached_prediction(gene_id: str, chem_id: str) -> dict:
    """
    Runs and caches the full prediction logic for a single pair.
    
    We pass in strings (gene_id, chem_id) which are hashable.
    The unhashable model/enricher objects are loaded from inside
    using their own cached loaders.
    """
    
    # Load all cached resources
    enricher = get_cached_enricher()
    model = get_cached_dti_model()
    protein_featurizer = get_cached_protein_featurizer()
    config = get_cached_config()
    device = get_cached_device()

    # Run the core logic
    return await process_pair_logic(
        gene_id, chem_id, 
        enricher, protein_featurizer, model, 
        config, device
    )
