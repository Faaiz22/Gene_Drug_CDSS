import torch
import torch.nn as nn
from torch_geometric.data import Data
from captum.attr import IntegratedGradients

from src.models.dti_model import DTIModel

class DTIModelWrapper(nn.Module):
    """
    A wrapper for the DTIModel to make it compatible with Captum.
    Captum's attribute methods require a forward function that accepts
    tensors as inputs, not a Data object.
    """
    def __init__(self, model: DTIModel):
        super().__init__()
        self.model = model

    def forward(self, x, edge_attr, protein_emb, edge_index, pos, batch):
        """
        Reconstructs the graph Data object from tensors and runs the model.
        """
        # Note: We create a minimal Data object. If your model needs more
        # attributes from the graph, they must be passed here.
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, batch=batch)
        
        # We only care about the logits, not attn_weights, for attribution
        logits, _ = self.model(graph, protein_emb)
        return logits

class ModelExplainer:
    """
    Handles the computation of model attributions using Integrated Gradients.
    """
    def __init__(self, model: DTIModel, device: torch.device):
        self.model = model
        self.device = device
        # Create the Captum-compatible wrapper
        self.model_wrapper = DTIModelWrapper(model).to(device)
        self.model_wrapper.eval()
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(self.model_wrapper)

    def explain(self, graph, protein_emb):
        """
        Calculates attributions for a single drug-protein pair.
        
        Args:
            graph (torch_geometric.data.Data): The input graph for the drug.
            protein_emb (torch.Tensor): The input protein embedding.

        Returns:
            A tuple of (atom_attributions, protein_attributions)
        """
        # Ensure inputs are on the correct device
        graph = graph.to(self.device)
        protein_emb = protein_emb.to(self.device)
        
        # 1. Define inputs for attribution
        inputs = (graph.x, graph.edge_attr, protein_emb)
        
        # 2. Define baselines (uninformative inputs, usually zeros)
        baselines = (
            torch.zeros_like(graph.x),
            torch.zeros_like(graph.edge_attr),
            torch.zeros_like(protein_emb)
        )
        
        # 3. Define additional forward args (static parts of the graph)
        additional_forward_args = (graph.edge_index, graph.pos, graph.batch)
        
        # 4. Compute attributions
        # n_steps is a tradeoff: higher is more accurate but slower.
        print("Calculating attributions... This may take a moment.")
        attributions = self.ig.attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=0,  # Target is the 0-th (and only) output logit
            n_steps=50,
            internal_batch_size=1
        )
        print("Attribution calculation complete.")

        # 5. Process and return the results
        Attributions[0] = atom features (x)
        Attributions[1] = edge features (edge_attr)
        Attributions[2] = protein embedding (protein_emb)
        
        # For atoms, we sum attributions across all features for each atom
        atom_attributions = attributions[0].sum(dim=1).cpu().numpy()
        
        # For protein, we sum attributions across the embedding dim
        # The embedding is [1, D], so we squeeze it.
        protein_attributions = attributions[2].sum(dim=1).squeeze(0).cpu().numpy()
        
        return atom_attributions, protein_attributions
