import unittest
import yaml
import pandas as pd
import torch
import os
from pathlib import Path

# Ensure we are in the project root for config loading
# This makes the test runnable from any directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'config/config.yaml'


try:
    from src.models.dti_model import DTIModel
    from src.preprocessing.feature_engineer import ProteinEmbedder
    from torch_geometric.data import Batch
    MODELS_LOADED = True
except ImportError as e:
    print(f"Skipping test_pipeline.py: Failed to import modules. Error: {e}")
    MODELS_LOADED = False


# Skip the entire test class if modules failed to load
@unittest.skipIf(not MODELS_LOADED, "Core model components not installed or found")
class TestPipelineComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load config once for all tests."""
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
            
        with open(CONFIG_PATH, 'r') as f:
            cls.config = yaml.safe_load(f)
            
        # Override device to CPU for testing to avoid GPU dependency
        cls.config['training']['device'] = 'cpu'
        cls.config['processing']['device'] = 'cpu'

    def test_dti_model_instantiation(self):
        """Test if the DTIModel can be created without errors."""
        try:
            model = DTIModel(self.config)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"DTIModel instantiation failed with error: {e}")

    def test_protein_embedder_instantiation(self):
        """Test if the ProteinEmbedder can be created without errors."""
        # This test will download the ESM model from Hugging Face if not cached
        # which is fine for an integration test.
        try:
            embedder = ProteinEmbedder(self.config)
            self.assertIsNotNone(embedder)
        except Exception as e:
            self.fail(f"ProteinEmbedder instantiation failed with error: {e}")

    def test_model_forward_pass_cpu(self):
        """Test a single forward pass with dummy data on CPU."""
        model = DTIModel(self.config)
        model.to(self.config['training']['device'])
        model.eval() # Set to evaluation mode

        # Create dummy data
        batch_size = 2
        num_atoms = 10
        
        # Ensure dimensions match config
        drug_node_dim = self.config['model']['drug_node_dim']
        drug_edge_dim = self.config['model']['drug_edge_dim']
        protein_emb_dim = self.config['model']['protein_embedding_dim']

        graph_dummy_list = []
        for _ in range(batch_size):
            num_atoms_i = torch.randint(5, 15, (1,)).item()
            num_edges_i = torch.randint(10, 20, (1,)).item()
            graph_dummy_list.append(
                Batch(
                    x=torch.randn(num_atoms_i, drug_node_dim),
                    edge_index=torch.randint(0, num_atoms_i, (2, num_edges_i)),
                    edge_attr=torch.randn(num_edges_i, drug_edge_dim),
                    pos=torch.randn(num_atoms_i, 3)
                )
            )
        
        # PyG Batch object handles the 'batch' tensor automatically
        graph_batch = Batch.from_data_list(graph_dummy_list)
        protein_dummy = torch.randn(batch_size, protein_emb_dim)
        
        graph_batch = graph_batch.to(self.config['training']['device'])
        protein_dummy = protein_dummy.to(self.config['training']['device'])

        try:
            with torch.no_grad(): # Disable gradient calculation
                logits, attn_weights = model(graph_batch, protein_dummy)
            
            self.assertEqual(logits.shape, (batch_size,))
            self.assertIsNotNone(attn_weights)
            # Check attention shape (batch_size, num_protein_tokens, 1)
            # Assuming protein_tokens = 1 after pooling, which it is in your model
            self.assertEqual(attn_weights.shape, (batch_size, 1, 1)) 
            
        except Exception as e:
            self.fail(f"Model forward pass failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
