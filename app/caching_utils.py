# app/caching_utils.py (COMPLETE REPLACEMENT)
import streamlit as st
import torch
from typing import Dict, Any
import asyncio

# Import all core logic functions
from src.core_processing import (
    get_device,
    load_config,
    load_protein_featurizer,
    load_dti_model,
)

# --- Cached Resource Loaders ---

@st.cache_resource
def get_cached_device() -> torch.device:
    """Caches the torch device."""
    return get_device()


@st.cache_resource
def get_cached_config() -> Dict[str, Any]:
    """Caches the app config."""
    return load_config()


@st.cache_resource
def get_cached_dti_model():
    """Caches the trained DTI model."""
    config = get_cached_config()
    device = get_cached_device()
    return load_dti_model(config, device)


@st.cache_resource
def get_cached_protein_featurizer():
    """Caches the protein language model."""
    config = get_cached_config()
    device = get_cached_device()
    return load_protein_featurizer(config, device)


# --- DO NOT CACHE ENRICHER - IT CONTAINS ASYNC CLIENT ---
# Create new enricher for each request
def get_enricher():
    """Creates a new DataEnricher instance (not cached due to async client)."""
    from src.utils.api_clients import DataEnricher
    config = get_cached_config()
    return DataEnricher(config)


# --- Cached Data/Prediction Function ---

@st.cache_data(ttl=3600 * 6, max_entries=500, show_spinner=False)
def get_cached_prediction(gene_id: str, chem_id: str) -> Dict[str, Any]:
    """
    Runs and caches the full prediction logic for a single pair.
    
    Args:
        gene_id: Gene identifier
        chem_id: Chemical identifier
    
    Returns:
        Dictionary with prediction results
    """
    # Import here to avoid circular dependency
    from src.core_processing import process_pair_logic_sync
    
    # Load all cached resources
    model = get_cached_dti_model()
    protein_featurizer = get_cached_protein_featurizer()
    config = get_cached_config()
    device = get_cached_device()
    
    # Create new enricher (not cached)
    enricher = get_enricher()

    # Run the core logic (synchronous wrapper)
    try:
        result = process_pair_logic_sync(
            gene_id, chem_id, 
            enricher, protein_featurizer, model, 
            config, device
        )
        return result
    except Exception as e:
        return {
            "gene_id": gene_id,
            "chem_id": chem_id,
            "error": str(e),
            "probability": None
        }
