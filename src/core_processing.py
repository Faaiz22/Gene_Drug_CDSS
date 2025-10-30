"""
CoreProcessor: The central engine for the CDSS.

This class orchestrates data fetching, preprocessing, and model inference.
It is initialized once and stored in the Streamlit session state.
"""

import torch
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

# --- Module Imports ---
# Import utility functions and classes
from .utils.config_loader import load_config  # Expose for other pages
from .utils.exceptions import CDSSException, DataFetchException, ModelException
from .utils.api_clients import PubChemClient, UniProtClient
from .utils.pubmed_client import PubMedClient

# Import core ML components
from .models.dti_model import DTIPredictor # Assuming this is the main model class
from .preprocessing.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

class CoreProcessor:
    """
    Orchestrates all backend logic for the CDSS application.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes all clients, featurizers, and the prediction model.
        
        Args:
            config: The application configuration dictionary.
        
        Raises:
            CDSSException: If any critical component fails to initialize.
        """
        logger.info("Initializing CoreProcessor...")
        
        # -----------------------------------------------------------------
        # THE FIX: Store the config object as a class attribute.
        # This resolves Error #1 (AttributeError).
        self.config = config
        # -----------------------------------------------------------------

        # 1. Initialize API Clients
        try:
            api_config = config['api']
            self.pubchem_client = PubChemClient(api_config.get('pubchem_base_url'))
            self.uniprot_client = UniProtClient(api_config.get('uniprot_base_url'))
            
            # PubMedClient uses the email/key from the config,
            # which were resolved from env vars by load_config.
            self.pubmed_client = PubMedClient(api_config['pubmed'])
            logger.info("API clients initialized.")
        except KeyError as e:
            raise CDSSException(f"Missing API configuration key: {e}", "Check config.yaml > api")
        except Exception as e:
            raise CDSSException(f"Failed to initialize API clients: {e}", "Check API configs and credentials")

        # 2. Initialize Feature Engineer
        try:
            self.feature_engineer = FeatureEngineer(config['model']['featurization'])
            logger.info("FeatureEngineer initialized.")
        except Exception as e:
            raise CDSSException(f"Failed to initialize FeatureEngineer: {e}", "Check config.yaml > model > featurization")

        # 3. Load the DTI Prediction Model
        try:
            self.model = self._load_model(config['model'])
            self.model.eval()  # Set model to evaluation mode
            logger.info("CoreProcessor initialized successfully.")
            
        except KeyError as e:
            raise CDSSException(f"Missing model configuration key: {e}", "Check config.yaml > model")
        except Exception as e:
            # Catch exceptions from _load_model (e.g., FileNotFoundError)
            raise CDSSException(f"Failed to load DTI model: {e}", str(e))

    def _load_model(self, model_config: Dict[str, Any]) -> torch.nn.Module:
        """
        Helper function to instantiate and load the model state dictionary.
        """
        model_path = model_config['model_path']
        
        # This addresses Error #5 (Missing .pt file)
        if not Path(model_path).is_file():
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(
                f"Model file (.pt) not found at path: '{model_path}'. "
                "Please upload your trained model and update config.yaml."
            )
        
        # Get model hyperparameters from config
        params = model_config.get('hyperparameters', {})
        
        # Instantiate model with parameters from config
        # This must match your model's __init__ signature
        try:
            model = DTIPredictor(
                protein_emb_dim=params.get('protein_emb_dim', 1280),
                drug_emb_dim=params.get('drug_emb_dim', 128),
                egnn_hidden_dim=params.get('egnn_hidden_dim', 128),
                # Add any other required model parameters here
            )
        except TypeError as e:
            raise ModelException(
                f"Model hyperparameter mismatch. Config keys may not match DTIPredictor __init__ args: {e}",
                "Check config.yaml > model > hyperparameters"
            )
        except ImportError:
            raise ModelException("Failed to import 'DTIPredictor' from 'models.dti_model'.", "Check the file and class name.")

        # Load the saved weights (state dictionary)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        logger.info(f"Model loaded from {model_path} and moved to {device}")
        return model

    # --- Synchronous Wrappers for Agent Tools ---
    # The agent tools run in a sync context, so we wrap our async API calls.

    def get_smiles_sync(self, drug_name: str) -> str:
        """Synchronous wrapper for async get_smiles_from_name."""
        try:
            return asyncio.run(self.pubchem_client.get_smiles_from_name(drug_name))
        except Exception as e:
            logger.error(f"Failed to get SMILES for {drug_name}: {e}")
            raise DataFetchException(f"Could not fetch SMILES for '{drug_name}'", str(e))

    def get_sequence_sync(self, gene_name: str) -> str:
        """Synchronous wrapper for async get_protein_sequence."""
        try:
            organism = self.config.get('api', {}).get('uniprot_organism', 'homo sapiens')
            query = f"gene:{gene_name} AND organism:\"{organism}\""
            return asyncio.run(self.uniprot_client.get_sequence_from_query(query))
        except Exception as e:
            logger.error(f"Failed to get sequence for {gene_name}: {e}")
            raise DataFetchException(f"Could not fetch protein sequence for '{gene_name}'", str(e))

    def get_literature_sync(self, drug_name: str, gene_name: str, max_results: int = 3) -> list:
        """Synchronous wrapper for async pubmed search."""
        try:
            return asyncio.run(self.pubmed_client.search_interactions(
                gene_name=gene_name,
                drug_name=drug_name,
                max_results=max_results
            ))
        except Exception as e:
            logger.error(f"Failed to get literature for {drug_name}/{gene_name}: {e}")
            raise DataFetchException(f"Could not fetch literature for '{drug_name} and {gene_name}'", str(e))

    # --- Core Pipeline Methods ---

    def get_preprocessed_data_for_pair(self, gene_id: str, chem_id: str):
        """
        Fetches data and runs preprocessing for the explanation page.
        Returns the data object required by the explainer.
        """
        try:
            # 1. Fetch data
            smiles = self.get_smiles_sync(chem_id)
            sequence = self.get_sequence_sync(gene_id)
            
            # 2. Featurize
            data = self.feature_engineer.featurize(smiles, sequence)
            
            # 3. Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            return data
            
        except Exception as e:
            logger.error(f"Failed to preprocess data for {gene_id}/{chem_id}: {e}", exc_info=True)
            raise ModelException(f"Failed to preprocess data for {gene_id}/{chem_id}", str(e))

    def run_model(self, smiles: str, sequence: str) -> Dict[str, Any]:
        """
        Runs the full DTI prediction pipeline on a single pair.
        """
        try:
            # 1. Featurize
            logger.debug("Featurizing inputs...")
            data = self.feature_engineer.featurize(smiles, sequence)
            
            # 2. Move to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            
            # 3. Predict
            logger.debug("Running model prediction...")
            with torch.no_grad():
                output = self.model(data)  # Model's forward pass
            
            # 4. Post-process output
            # This logic assumes the model output needs sigmoid (binary classification)
            if output.dim() > 1 and output.shape[1] == 2:
                # Binary classification logits [batch_size, 2]
                prob = torch.softmax(output, dim=1)[0][1].item() # Prob of class 1 (binds)
            elif output.dim() == 1 or output.shape[1] == 1:
                # Single logit or probability [batch_size, 1]
                prob = torch.sigmoid(output)[0].item()
            else:
                raise ModelException(f"Unexpected model output shape: {output.shape}")

            threshold = self.config.get('model', {}).get('threshold', 0.5)
            prediction = 1 if prob > threshold else 0
            
            logger.info(f"Prediction complete for {smiles[:10]}.../{sequence[:10]}... -> Prob: {prob:.4f}, Class: {prediction}")
            
            return {
                "probability": prob,
                "prediction": prediction,
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Model prediction pipeline failed: {e}", exc_info=True)
            raise ModelException("Model prediction failed", str(e))

