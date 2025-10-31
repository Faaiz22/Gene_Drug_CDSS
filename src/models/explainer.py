"""
Enhanced Explainability Module with Integrated Gradients,
Generative Explanations, and Advanced Visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

from dti_model import DTIModel

logger = logging.getLogger(__name__)


class DTIModelWrapper(nn.Module):
    """
    Captum-compatible wrapper for DTIModel.
    Enables attribution methods on graph inputs.
    """
    def __init__(self, model: DTIModel):
        super().__init__()
        self.model = model
    
    def forward(self, x, edge_attr, protein_emb, edge_index, pos, batch):
        """
        Reconstructs graph and runs model forward pass.
        """
        from torch_geometric.data import Data
        
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            batch=batch
        )
        
        logits, _ = self.model(graph, protein_emb)
        return logits


class EnhancedExplainer:
    """
    Comprehensive explainability system for DTI predictions.
    """
    
    def __init__(self, model: DTIModel, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.xai_config = config['explainability']
        
        # Wrap model for Captum
        self.model_wrapper = DTIModelWrapper(model).to(device)
        self.model_wrapper.eval()
        
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(self.model_wrapper)
        
        # Load generative model for NL explanations
        self.text_generator = None
        self.text_tokenizer = None
        
        if self.xai_config['use_generative_explanations']:
            try:
                gen_model = self.xai_config['generative_model']
                self.text_tokenizer = AutoTokenizer.from_pretrained(gen_model)
                self.text_generator = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(device)
                self.text_generator.eval()
                logger.info(f"Loaded generative model: {gen_model}")
            except Exception as e:
                logger.warning(f"Failed to load generative model: {e}")
    
    def explain_prediction(
        self,
        graph,
        protein_emb: torch.Tensor,
        gene_name: str,
        drug_name: str,
        smiles: str,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction.
        
        Returns:
            Dictionary with:
            - prediction_prob: float
            - atom_attributions: np.ndarray
            - protein_attributions: np.ndarray
            - top_atoms: List[int]
            - top_residues: List[int]
            - natural_language: str
            - visualization_paths: Dict[str, Path]
        """
        logger.info(f"Generating explanation for {drug_name} - {gene_name}")
        
        # Move inputs to device
        graph = graph.to(self.device)
        protein_emb = protein_emb.to(self.device)
        
        # 1. Get prediction
        with torch.no_grad():
            logits, attn_weights = self.model(graph, protein_emb.unsqueeze(0))
            prob = torch.sigmoid(logits).item()
        
        logger.info(f"Prediction probability: {prob:.4f}")
        
        # 2. Compute Integrated Gradients
        atom_attrs, edge_attrs, protein_attrs = self._compute_attributions(
            graph, protein_emb
        )
        
        # 3. Identify key features
        top_atoms = self._get_top_features(atom_attrs, k=5)
        top_residues = self._get_top_features(protein_attrs, k=10)
        
        # 4. Generate natural language explanation
        nl_explanation = self._generate_nl_explanation(
            gene_name, drug_name, prob,
            top_atoms, top_residues, atom_attrs, protein_attrs
        )
        
        # 5. Create visualizations
        viz_paths = {}
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Molecular structure with atom importance
            mol_path = save_dir / f"{drug_name}_atom_importance.png"
            self._visualize_molecule(smiles, atom_attrs, mol_path)
            viz_paths['molecule'] = mol_path
            
            # Attention heatmap
            if attn_weights is not None:
                attn_path = save_dir / f"{drug_name}_{gene_name}_attention.png"
                self._visualize_attention(attn_weights, attn_path)
                viz_paths['attention'] = attn_path
            
            # Attribution distribution
            dist_path = save_dir / f"{drug_name}_{gene_name}_attribution_dist.png"
            self._visualize_attribution_distribution(
                atom_attrs, protein_attrs, dist_path
            )
            viz_paths['distribution'] = dist_path
        
        return {
            'prediction_prob': prob,
            'atom_attributions': atom_attrs,
            'protein_attributions': protein_attrs,
            'top_atoms': top_atoms,
            'top_residues': top_residues,
            'natural_language': nl_explanation,
            'visualization_paths': viz_paths,
            'attention_weights': attn_weights.cpu().numpy() if attn_weights is not None else None
        }
    
    def _compute_attributions(
        self,
        graph,
        protein_emb: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute feature attributions using Integrated Gradients.
        """
        # Prepare inputs
        inputs = (graph.x, graph.edge_attr, protein_emb.unsqueeze(0))
        
        # Define baselines
        baseline_strategy = self.xai_config['ig_baseline_strategy']
        
        if baseline_strategy == "zero":
            baselines = (
                torch.zeros_like(graph.x),
                torch.zeros_like(graph.edge_attr),
                torch.zeros_like(protein_emb.unsqueeze(0))
            )
        elif baseline_strategy == "gaussian":
            baselines = (
                torch.randn_like(graph.x) * 0.1,
                torch.randn_like(graph.edge_attr) * 0.1,
                torch.randn_like(protein_emb.unsqueeze(0)) * 0.1
            )
        else:  # uniform
            baselines = (
                torch.rand_like(graph.x),
                torch.rand_like(graph.edge_attr),
                torch.rand_like(protein_emb.unsqueeze(0))
            )
        
        # Additional forward arguments (static graph structure)
        additional_args = (graph.edge_index, graph.pos, graph.batch)
        
        # Compute attributions
        logger.info("Computing Integrated Gradients...")
        attributions = self.ig.attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=additional_args,
            target=0,
            n_steps=self.xai_config['ig_steps'],
            internal_batch_size=1
        )
        
        # Process attributions
        # Sum across feature dimensions for interpretability
        atom_attrs = attributions[0].sum(dim=1).cpu().numpy()
        edge_attrs = attributions[1].sum(dim=1).cpu().numpy()
        protein_attrs = attributions[2].sum(dim=2).squeeze(0).cpu().numpy()
        
        # Normalize to [0, 1] for visualization
        atom_attrs = self._normalize_attributions(atom_attrs)
        protein_attrs = self._normalize_attributions(protein_attrs)
        
        logger.info("Attribution computation complete")
        
        return atom_attrs, edge_attrs, protein_attrs
    
    @staticmethod
    def _normalize_attributions(attrs: np.ndarray) -> np.ndarray:
        """Normalize attributions to [0, 1] range."""
        attrs_abs = np.abs(attrs)
        if attrs_abs.max() > 0:
            return attrs_abs / attrs_abs.max()
        return attrs_abs
    
    @staticmethod
    def _get_top_features(attrs: np.ndarray, k: int) -> List[int]:
        """Get indices of top-k most important features."""
        return np.argsort(attrs)[-k:][::-1].tolist()
    
    def _generate_nl_explanation(
        self,
        gene_name: str,
        drug_name: str,
        prob: float,
        top_atoms: List[int],
        top_residues: List[int],
        atom_attrs: np.ndarray,
        protein_attrs: np.ndarray
    ) -> str:
        """
        Generate natural language explanation using a generative model.
        """
        # Construct prompt for the generative model
        prompt = f"""Explain why the drug {drug_name} {"likely interacts with" if prob > 0.5 else "likely does not interact with"} the protein {gene_name}.

Prediction confidence: {prob:.2%}

Key contributing drug atoms (by index): {top_atoms[:3]}
Key protein regions (residue indices): {top_residues[:5]}

The model identified {len([a for a in atom_attrs if a > 0.5])} highly important atoms in the drug molecule.
The protein analysis highlighted {len([p for p in protein_attrs if p > 0.5])} critical residues.

Provide a clear, scientifically accurate explanation in 2-3 sentences suitable for a clinician."""
        
        if self.text_generator and self.text_tokenizer:
            try:
                inputs = self.text_tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                outputs = self.text_generator.generate(
                    **inputs,
                    max_length=self.xai_config['generative_max_length'],
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
                
                explanation = self.text_tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                return explanation
            
            except Exception as e:
                logger.warning(f"Generative explanation failed: {e}")
        
        # Fallback template-based explanation
        if prob > 0.7:
            strength = "strong"
        elif prob > 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        
        explanation = (
            f"The model predicts a {strength} interaction between {drug_name} and {gene_name} "
            f"with {prob:.1%} confidence. "
            f"The prediction is primarily driven by {len(top_atoms)} key atoms in the drug molecule "
            f"and {len(top_residues)} critical residues in the protein. "
            f"Atoms at positions {top_atoms[:3]} show the highest contribution to binding affinity."
        )
        
        return explanation
    
    def _visualize_molecule(
        self,
        smiles: str,
        atom_attrs: np.ndarray,
        save_path: Path
    ):
        """
        Visualize molecule with atom importances highlighted.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create color map
        cmap = plt.get_cmap(self.xai_config['colormap'])
        colors = {}
        
        for i in range(mol.GetNumAtoms()):
            if i < len(atom_attrs):
                # Map attribution to color
                colors[i] = cmap(atom_attrs[i])[:3]  # RGB only
        
        # Draw molecule with highlighted atoms
        drawer = Draw.MolDraw2DCairo(800, 800)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(min(mol.GetNumAtoms(), len(atom_attrs)))),
            highlightAtomColors=colors
        )
        drawer.FinishDrawing()
        
        # Save image
        with open(save_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        logger.info(f"Molecule visualization saved to {save_path}")
    
    def _visualize_attention(
        self,
        attn_weights: torch.Tensor,
        save_path: Path
    ):
        """
        Visualize cross-attention weights as a heatmap.
        """
        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
        # Average across heads
        attn = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.xlabel('Drug Atoms')
        plt.ylabel('Protein Residues')
        plt.title('Cross-Attention Weights: Protein → Drug')
        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=self.xai_config['visualization_dpi'],
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"Attention heatmap saved to {save_path}")
    
    def _visualize_attribution_distribution(
        self,
        atom_attrs: np.ndarray,
        protein_attrs: np.ndarray,
        save_path: Path
    ):
        """
        Visualize distribution of attributions for drug and protein.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Drug atom attributions
        ax1.hist(atom_attrs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(atom_attrs.mean(), color='red', linestyle='--', 
                    label=f'Mean: {atom_attrs.mean():.3f}')
        ax1.set_xlabel('Attribution Magnitude')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Drug Atom Attribution Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Protein residue attributions
        ax2.hist(protein_attrs, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.axvline(protein_attrs.mean(), color='blue', linestyle='--',
                    label=f'Mean: {protein_attrs.mean():.3f}')
        ax2.set_xlabel('Attribution Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Protein Residue Attribution Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=self.xai_config['visualization_dpi'],
            bbox_inches='tight'
        )
        plt.close()
        
        logger.info(f"Attribution distribution plot saved to {save_path}")
