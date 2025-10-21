import unittest
import yaml
import pandas as pd
import torch

# It's better to test individual components, but for a start,
# we can create a simple test to check if the main classes can be instantiated.

from src.models.dti_model import DTIModel
from src.preprocessing.feature_engineer import ProteinEmbedder

class TestPipelineComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load a mock config
        with open('config/config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
        # Use CPU for testing to avoid GPU dependency
        cls.config['training']['device'] = 'cpu'

    def test_dti_model_instantiation(self):
        """Test if the DTIModel can be created without errors."""
        try:
            model = DTIModel(self.config)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"DTIModel instantiation failed with error: {e}")

    def test_protein_embedder_instantiation(self):
        """Test if the ProteinEmbedder can be created without errors."""
        # This test will download the model from Hugging Face if not cached
        try:
            embedder = ProteinEmbedder(self.config)
            self.assertIsNotNone(embedder)
        except Exception as e:
            self.fail(f"ProteinEmbedder instantiation failed with error: {e}")

    def test_model_forward_pass(self):
        """Test a single forward pass with dummy data."""
        model = DTIModel(self.config)
        model.to(self.config['training']['device'])

        # Create dummy data
        batch_size = 2
        num_atoms = 10
        graph_dummy = {
            'x': torch.randn(num_atoms * batch_size, self.config['model']['drug_node_dim']),
            'edge_index': torch.randint(0, num_atoms * batch_size, (2, 20)),
            'edge_attr': torch.randn(20, self.config['model']['drug_edge_dim']),
            'pos': torch.randn(num_atoms * batch_size, 3),
            'batch': torch.repeat_interleave(torch.arange(batch_size), num_atoms)
        }
        protein_dummy = torch.randn(batch_size, self.config['model']['protein_embedding_dim'])

        from torch_geometric.data import Batch
        graph_batch = Batch(**graph_dummy)

        try:
            logits, attn_weights = model(graph_batch, protein_dummy)
            self.assertEqual(logits.shape, (batch_size,))
            self.assertIsNotNone(attn_weights)
        except Exception as e:
            self.fail(f"Model forward pass failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
